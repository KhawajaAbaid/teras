import jax
import keras
from keras import random, ops
from teras.backend.jax.models.gans.jaxgan import JAXGAN
from teras.backend.common.models.gans.ctgan.ctgan import BaseCTGAN


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
        x_generated = self.generator.stateless_call(
            generator_trainable_vars,
            generator_non_trainable_vars,
            noise)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        y_pred_generated = self.discriminator.stateless_call(
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
        y_pred_generated = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_generated)
        y_pred_real = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x)
        grad_pen = self.discriminator.gradient_penalty(x, x_generated)
        return grad_pen, (discriminator_non_trainable_vars,)

    def discriminator_compute_loss_and_updates(
            self,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x,
            x_generated,
    ):
        y_pred_generated = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_generated)
        y_pred_real = self.discriminator.stateless_call(
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

        # Get generator state
        # Since generator comes and gets built before discriminator
        generator_trainable_vars = trainable_variables[
                                   :len(self.generator.trainable_variables)]
        generator_non_trainable_vars = non_trainable_variables[
                                       :len(self.generator.non_trainable_variables)]
        generator_optimizer_vars = optimizer_variables[
                                   :len(self.generator_optimizer.variables)]

        # Get discriminator state
        discriminator_trainable_vars = trainable_variables[
                                       len(self.generator.trainable_variables):]
        discriminator_non_trainable_vars = non_trainable_variables[
                                           len(self.generator.non_trainable_variables):]
        discriminator_optimizer_vars = optimizer_variables[
                                       len(self.generator_optimizer.variables):
                                       ]

        x, cond_vectors_real, cond_vectors, mask = data
        batch_size = ops.shape(x)[0]

        # =========================
        # Discriminator train step
        # =========================
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
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
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
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

        state = (
            generator_trainable_vars + discriminator_trainable_vars,
            generator_non_trainable_vars + discriminator_non_trainable_vars,
            generator_optimizer_vars + discriminator_optimizer_vars,
            new_metric_variables
        )
        return logs, state
