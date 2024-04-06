import jax
import keras
from keras import random, ops
from keras.backend import floatx
from teras.backend.jax.models.gans.jaxgan import JAXGAN
from teras.backend.common.models.gans.pcgain import BasePCGAIN


class PCGAIN(JAXGAN, BasePCGAIN):
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 classifier: keras.Model,
                 hint_rate: float = 0.9,
                 alpha: float = 200.,
                 beta: float = 100.,
                 **kwargs
                 ):
        JAXGAN.__init__(self,
                        generator=generator,
                        discriminator=discriminator)
        BasePCGAIN.__init__(self,
                            generator=generator,
                            discriminator=discriminator,
                            classifier=classifier,
                            hint_rate=hint_rate,
                            alpha=alpha,
                            beta=beta,
                            **kwargs)

    def generator_compute_loss_and_updates(
            self,
            generator_trainable_vars,
            generator_non_trainable_vars,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            classifier_trainable_vars,
            classifier_non_trainable_vars,
            x_gen,
            hint_vectors,
            mask,
            training=False
    ):
        x_generated, generator_non_trainable_vars = self.generator.stateless_call(
            generator_trainable_vars,
            generator_non_trainable_vars,
            ops.concatenate([x_gen, mask], axis=1),
            training=training,
        )
        x_hat = (x_generated * (1 - mask)) + (x_gen * mask)
        classifier_pred, _ = self.classifier.stateless_call(
            classifier_trainable_vars,
            classifier_non_trainable_vars,
            x_hat
        )
        mask_pred, _ = (
            self.discriminator.stateless_call(
                discriminator_trainable_vars,
                discriminator_non_trainable_vars,
                ops.concatenate([x_hat, hint_vectors], axis=1),
                training=training,
            )
        )
        loss = self.compute_generator_loss(
            x=x_gen,
            x_generated=x_generated,
            mask=mask,
            mask_pred=mask_pred,
            classifier_pred=classifier_pred,
            alpha=self.alpha,
            beta=self.beta,
        )
        return loss, (generator_non_trainable_vars,)

    def discriminator_compute_loss_and_updates(
            self,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_hat_disc,
            hint_vectors,
            mask,
            training=False
    ):
        (
            mask_pred,
            discriminator_non_trainable_vars
        ) = self.discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            ops.concatenate([x_hat_disc, hint_vectors], axis=1),
            training=training,
        )
        loss = self.compute_discriminator_loss(mask, mask_pred)
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
        disc_slice_start_idx = len(self.generator.trainable_variables)
        disc_slice_end_idx = disc_slice_start_idx + len(
            self.discriminator.trainable_variables)
        discriminator_trainable_vars = trainable_variables[
                                    disc_slice_start_idx: disc_slice_end_idx]
        discriminator_non_trainable_vars = non_trainable_variables[
                                    disc_slice_start_idx: disc_slice_end_idx]
        discriminator_optimizer_vars = optimizer_variables[
                                       len(self.generator_optimizer.variables):]

        # Get classifier state
        classifier_trainable_vars = trainable_variables[
                                    -len(self.classifier.trainable_variables):]
        classifier_non_trainable_vars = non_trainable_variables[
                                        -len(self.classifier.non_trainable_variables):]

        # data is a tuple of x_generator and x_discriminator batches
        # drawn from the dataset. The reason behind generating two separate
        # batches of data at each step is that it's how GAIN's algorithm works
        x_gen, x_disc = data

        # =========================
        # Train the discriminator
        # =========================
        # Create mask
        mask = 1. - ops.cast(ops.isnan(x_disc), dtype=floatx())
        # replace nans with 0.
        x_disc = ops.where(ops.isnan(x_disc), x1=0., x2=x_disc)
        # Sample noise
        z = random.uniform(shape=ops.shape(x_disc), minval=0., maxval=0.01,
                           seed=1337)
        # Sample hint vectors
        hint_vectors = random.binomial(shape=ops.shape(x_disc),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=1337)
        hint_vectors = hint_vectors * mask
        # Combine random vectors with original data
        x_disc = x_disc * mask + (1 - mask) * z
        x_generated, _ = self.generator.stateless_call(
            generator_trainable_vars,
            generator_non_trainable_vars,
            ops.concatenate([x_disc, mask], axis=1),
        )
        x_hat_disc = (x_generated * (1 - mask)) + (x_disc * mask)
        disc_grad_fn = jax.value_and_grad(
            self.discriminator_compute_loss_and_updates,
            has_aux=True,
        )
        (d_loss, (discriminator_non_trainable_vars,)), grads = disc_grad_fn(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_hat_disc,
            hint_vectors,
            mask,
            training=True,
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
        # Train the generator
        # =====================
        mask = 1. - ops.cast(ops.isnan(x_gen), dtype=floatx())
        x_gen = ops.where(ops.isnan(x_gen), x1=0., x2=x_gen)
        z = random.uniform(shape=ops.shape(x_gen), minval=0., maxval=0.01,
                           seed=1337)
        hint_vectors = random.binomial(shape=ops.shape(x_gen),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=1337)
        hint_vectors = hint_vectors * mask
        x_gen = x_gen * mask + (1 - mask) * z

        gen_grad_fn = jax.value_and_grad(
            self.generator_compute_loss_and_updates,
            has_aux=True,
        )
        (g_loss, (generator_non_trainable_vars,)), grads = gen_grad_fn(
            generator_trainable_vars,
            generator_non_trainable_vars,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            classifier_trainable_vars,
            classifier_non_trainable_vars,
            x_gen,
            hint_vectors,
            mask,
            training=True,
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
            generator_trainable_vars + discriminator_trainable_vars +
            classifier_trainable_vars,
            generator_non_trainable_vars + discriminator_non_trainable_vars +
            classifier_non_trainable_vars,
            generator_optimizer_vars + discriminator_optimizer_vars,
            new_metric_variables
        )
        return logs, state