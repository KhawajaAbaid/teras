import jax
import keras
from keras import ops
from teras.backend.common.models.vaes.tvae.tvae import BaseTVAE


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
                                 sigmas=sigmas,
                                 mean=mean,
                                 log_var=log_var)
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
        metrics_variables = self.loss_tracker.statless_update_state(
            metrics_variables,
            loss)
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables
        )
        logs = {m.name: m.result() for m in self.metrics}
        return logs, state
