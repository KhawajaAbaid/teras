import jax
from jax import lax
import keras
from keras import ops
from teras._src.backend.common.models.autoencoders.tvae.tvae import BaseTVAE


class TVAE(BaseTVAE):
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 metadata: dict,
                 data_dim: int,
                 latent_dim: int = 128,
                 loss_factor: float = 2.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         metadata=metadata,
                         data_dim=data_dim,
                         latent_dim=latent_dim,
                         loss_factor=loss_factor,
                         seed=seed,
                         **kwargs)

    def compute_loss(self, real_samples=None, generated_samples=None,
                     sigmas=None, mean=None, log_var=None, metadata=None,
                     data_dim=None, loss_factor=None):
        loss = []
        cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="sum")
        cont_i = 0  # continuous index
        cat_i = 0  # categorical index
        for i, relative_index in enumerate(metadata["relative_indices_all"]):
            # the first k features are continuous
            if i < metadata["num_continuous"]:
                # each continuous feature is of the form
                # [alpha, beta1, beta2...beta(n)] where n is the number of
                # clusters

                # calculate alpha loss
                std = sigmas[relative_index]
                eq = real_samples[:, relative_index] - ops.tanh(
                    generated_samples[:, relative_index])
                loss_temp = ops.sum((eq ** 2 / 2 / (std ** 2)))
                loss.append(loss_temp)
                loss_temp = ops.log(std) * ops.cast(
                    ops.shape(real_samples)[0], dtype=std.dtype)
                loss.append(loss_temp)

                # calculate betas loss
                num_clusters = metadata["continuous"][
                    "num_valid_clusters_all"][cont_i]
                logits = lax.dynamic_slice_in_dim(
                    generated_samples,
                    start_index=relative_index + 1,
                    slice_size=num_clusters
                )
                labels = ops.argmax(
                    lax.dynamic_slice_in_dim(
                        real_samples,
                        start_index=relative_index + 1,
                        slice_size=num_clusters,
                    ),
                    axis=-1
                )
                cross_entropy_loss = cross_entropy(y_pred=logits, y_true=labels)
                loss.append(cross_entropy_loss)
                cont_i += 1
            else:
                num_categories = metadata["categorical"][
                    "num_categories_all"][cat_i]
                logits = lax.dynamic_slice_in_dim(
                    generated_samples,
                    start_index=relative_index,
                    slice_size=num_categories
                )
                labels = ops.argmax(
                    lax.dynamic_slice_in_dim(
                        real_samples,
                        start_index=relative_index,
                        slice_size=num_categories
                    ),
                    axis=-1
                )
                logits = ops.squeeze(logits)
                cross_entropy_loss = cross_entropy(y_pred=logits, y_true=labels)
                loss.append(cross_entropy_loss)
                cat_i += 1
        KLD = -0.5 * ops.sum(1 + log_var - mean ** 2 - ops.exp(log_var))
        loss_1 = ops.sum(loss) * loss_factor / data_dim
        loss_2 = KLD / data_dim
        final_loss = loss_1 + loss_2
        return final_loss

    def compute_loss_and_updates(self, trainable_variables,
                                 non_trainable_variables, x):
        (
            (generated_samples, sigmas, mean, log_var),
            non_trainable_variables
        ) = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
        )
        loss = self.compute_loss(real_samples=x,
                                 generated_samples=generated_samples,
                                 sigmas=sigmas, mean=mean, log_var=log_var,
                                 metadata=self.metadata,
                                 data_dim=self.data_dim,
                                 loss_factor=self.loss_factor)
        return loss, (non_trainable_variables, sigmas)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables
        ) = state
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates,
                                     has_aux=True)
        (loss, (non_trainable_variables, sigmas)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            data
        )
        (
            trainable_variables,
            optimizer_variables
        ) = self.optimizer.stateless_apply(optimizer_variables,
                                           grads,
                                           trainable_variables)
        sigmas = ops.clip(sigmas, x_min=0.01, x_max=1.0)
        self.decoder.sigmas = sigmas
        # since we only have one metric i.e. loss.
        metrics_variables = self.loss_tracker.stateless_update_state(
            metrics_variables,
            loss)
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables
        )
        logs = {"loss": self.loss_tracker.stateless_result(metrics_variables)}
        return logs, state

    def test_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state
        (loss,
         (non_trainable_variables, sigmas)
         ), grads = self.compute_loss_and_updates(
            trainable_variables,
            non_trainable_variables,
            data
        )
        # since we only have one metric i.e. loss.
        metrics_variables = self.loss_tracker.stateless_update_state(
            metrics_variables,
            loss)
        state = (
            trainable_variables,
            non_trainable_variables,
            metrics_variables
        )
        logs = {"loss": self.loss_tracker.stateless_result(metrics_variables)}
        return logs, state

    def predict_step(self, state, z):
        trainable_variables, non_trainable_variables = state
        # Since encoder comes hence gets built before decoder, so the parent
        # model's list of vars contains encoder vars followed by decoder vars.
        # if you didn't know that, feel free to thank me! jk. happy coding!
        start_idx = len(self.encoder.trainable_variables)
        slice_size = len(self.decoder.trainable_variables)
        decoder_trainable_vars = lax.dynamic_slice_in_dim(
            trainable_variables,
            start_index=start_idx,
            slice_size=slice_size
        )
        start_idx = len(self.encoder.non_trainable_variables)
        slice_size = len(self.decoder.non_trainable_variables)
        decoder_non_trainable_vars = lax.dynamic_slice_in_dim(
            non_trainable_variables,
            start_index=start_idx,
            slice_size=slice_size
        )
        (
            (generated_samples, sigmas),
            decoder_non_trainable_vars
        ) = self.decoder.stateless_call(
            decoder_trainable_vars,
            decoder_non_trainable_vars,
            z
        )
        return generated_samples, decoder_non_trainable_vars
