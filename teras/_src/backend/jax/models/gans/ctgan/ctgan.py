import jax
import keras
from keras import random, ops
from teras._src.backend.jax.models.gans.jaxgan import JAXGAN
from teras._src.backend.common.models.gans.ctgan.ctgan import BaseCTGAN


class CTGAN(JAXGAN, BaseCTGAN):
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 metadata: dict,
                 latent_dim: int = 128,
                 seed: int = 1337,
                 **kwargs):
        JAXGAN.__init__(self,
                        generator=generator,
                        discriminator=discriminator,
                        **kwargs)
        BaseCTGAN.__init__(self,
                           generator=generator,
                           discriminator=discriminator,
                           metadata=metadata,
                           latent_dim=latent_dim,
                           seed=seed)

    @staticmethod
    def compute_generator_loss(x_generated,
                               y_pred_generated,
                               cond_vectors,
                               mask,
                               metadata):
        """
        Loss for the Generator model in the CTGAN architecture.

        Args:
            x_generated: Samples drawn from the input dataset
            y_pred_generated: Discriminator's output for the generated samples
            cond_vectors: Conditional vectors that are used for and with
                generated samples
            mask: Mask created during the conditional vectors generation step
            metadata: dict, metadata computed during the data transformation step.

        Returns:
            Generator's loss.
        """
        loss = []
        cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None)
        continuous_features_relative_indices = metadata["continuous"][
            "relative_indices_all"]
        features_relative_indices_all = metadata["relative_indices_all"]
        num_categories_all = metadata["categorical"]["num_categories_all"]
        # the first k features in the data are continuous which we'll ignore as
        # we're only concerned with the categorical features here
        offset = len(continuous_features_relative_indices)
        for i, index in enumerate(features_relative_indices_all[offset:]):
            start_idx = index
            slice_size = num_categories_all[i]
            logits = jax.lax.dynamic_slice_in_dim(
                x_generated, start_index=start_idx,
                slice_size=slice_size, axis=1
            )
            start_idx = i
            slice_size = num_categories_all[i]
            temp_cond_vector = jax.lax.dynamic_slice_in_dim(
                cond_vectors, start_index=start_idx,
                slice_size=slice_size, axis=1
            )
            labels = ops.argmax(temp_cond_vector, axis=1)
            ce_loss = cross_entropy_loss(y_pred=logits,
                                         y_true=labels
                                         )
            loss.append(ce_loss)
        loss = ops.stack(loss, axis=1)
        loss = ops.sum(
            loss * ops.cast(mask, dtype="float32")
                       ) / ops.cast(ops.shape(y_pred_generated)[0],
                                    dtype="float32")
        loss = -ops.mean(y_pred_generated) * loss
        return loss

    def generator_compute_loss_and_updates(
            self,
            generator_trainable_vars,
            generator_non_trainable_vars,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            noise,
            cond_vectors,
            mask,
            metadata
    ):
        (
            x_generated,
            generator_non_trainable_vars
        ) = self.generator.stateless_call(
            generator_trainable_vars,
            generator_non_trainable_vars,
            noise)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        (
            y_pred_generated,
            discriminator_non_trainable_vars
        ) = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_generated)
        loss = self.compute_generator_loss(x_generated, y_pred_generated,
                                           cond_vectors=cond_vectors, mask=mask,
                                           metadata=metadata)
        return loss, (generator_non_trainable_vars,)

    def discriminator_compute_grad_penalty_and_updates(
            self,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x,
            x_generated,
    ):
        (
            y_pred_generated,
            discriminator_non_trainable_vars
        ) = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_generated)
        (
            y_pred_real,
            discriminator_non_trainable_vars
        ) = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x)
        (
            grad_pen,
            discriminator_non_trainable_vars
        ) = self.discriminator.gradient_penalty(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x, x_generated)
        return grad_pen, (discriminator_non_trainable_vars,)

    def discriminator_compute_loss_and_updates(
            self,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x,
            x_generated,
    ):
        (
            y_pred_generated,
            discriminator_non_trainable_vars
        ) = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_generated)
        (
            y_pred_real,
            discriminator_non_trainable_vars
        ) = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x)
        loss = self.compute_disciriminator_loss(y_pred_real, y_pred_generated)
        return loss, (discriminator_non_trainable_vars,)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables
        ) = state

        seed = non_trainable_variables[0]

        # Get generator state
        # Since generator comes and gets built before discriminator
        generator_trainable_vars = trainable_variables[
                                   :len(self.generator.trainable_variables)]
        generator_non_trainable_vars = non_trainable_variables[1:][
                                       :len(self.generator.non_trainable_variables)]
        generator_optimizer_vars = optimizer_variables[
                                   :len(self.generator_optimizer.variables)]

        # Get discriminator state
        discriminator_trainable_vars = trainable_variables[
                                       len(self.generator.trainable_variables):]
        discriminator_non_trainable_vars = non_trainable_variables[1:][
                                           len(self.generator.non_trainable_variables):]
        discriminator_optimizer_vars = optimizer_variables[
                                       len(self.generator_optimizer.variables):
                                       ]

        actual_data, dummy_y = data
        del dummy_y
        x, cond_vectors_real, cond_vectors, mask = actual_data
        batch_size = ops.shape(x)[0]

        # =========================
        # Discriminator train step
        # =========================
        seed = jax.random.split(seed, 1)[0]
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=seed)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)
        x_generated, _ = self.generator.stateless_call(
            generator_trainable_vars,
            generator_non_trainable_vars,
            input_gen)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        x = ops.concatenate([x, cond_vectors_real],
                            axis=1)
        # w.r.t. gradient penalty
        disc_grad_fn_pen = jax.value_and_grad(
            self.discriminator_compute_grad_penalty_and_updates,
            has_aux=True)
        (grad_pen,
         (discriminator_non_trainable_vars,)), grads = disc_grad_fn_pen(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x,
            x_generated
        )
        (
            discriminator_trainable_vars,
            discriminator_optimizer_vars
        ) = self.discriminator_optimizer.stateless_apply(
            discriminator_optimizer_vars,
            grads,
            discriminator_trainable_vars
        )
        # w.r.t. loss
        disc_grad_fn_loss = jax.value_and_grad(
            self.discriminator_compute_loss_and_updates,
            has_aux=True)
        (
            d_loss, (discriminator_non_trainable_vars,)), grads = disc_grad_fn_loss(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x,
            x_generated
        )
        (
            discriminator_trainable_vars,
            discriminator_optimizer_vars
        ) = self.discriminator_optimizer.stateless_apply(
            discriminator_optimizer_vars,
            grads,
            discriminator_trainable_vars
        )

        # =====================
        # Generator train step
        # =====================
        seed = jax.random.split(seed, 1)[0]
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=seed)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)
        gen_grad_fn = jax.value_and_grad(
            self.generator_compute_loss_and_updates,
            has_aux=True,
        )
        (g_loss, (generator_non_trainable_vars,)), grads = gen_grad_fn(
            generator_trainable_vars,
            generator_non_trainable_vars,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            input_gen,
            cond_vectors,
            mask,
            self.metadata,
        )

        (
            generator_trainable_vars,
            generator_optimizer_vars
        ) = self.generator_optimizer.stateless_apply(
            generator_optimizer_vars,
            grads,
            generator_trainable_vars
        )

        # Update metrics
        logs = {}
        new_metric_variables = []
        for metric in self.metrics:
            this_metric_variables = metrics_variables[
                                    len(new_metric_variables): len(new_metric_variables) + len(metric.variables)
                                    ]
            if metric.name == "generator_loss":
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    g_loss
                )
            elif metric.name == "discriminator_loss":
                this_metric_variables = metric.stateless_update_state(
                    this_metric_variables,
                    d_loss
                )
            else:
                continue
            logs[metric.name] = metric.stateless_result(this_metric_variables)
            new_metric_variables += this_metric_variables

        # Update seed
        seed = jax.random.split(seed, 1)[0]
        state = (
            generator_trainable_vars + discriminator_trainable_vars,
            [seed] + generator_non_trainable_vars +
            discriminator_non_trainable_vars,
            generator_optimizer_vars + discriminator_optimizer_vars,
            new_metric_variables
        )
        return logs, state
