import tensorflow as tf
import keras
from keras import random, ops
from keras.backend import floatx
from teras._src.backend.common.models.gans.pcgain import BasePCGAIN


class PCGAIN(BasePCGAIN):
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 classifier: keras.Model,
                 hint_rate: float = 0.9,
                 alpha: float = 200.,
                 beta: float = 100.,
                 **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         classifier=classifier,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         beta=beta,
                         **kwargs)

    def train_step(self, data):
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
                           seed=self._seed_gen)
        # Sample hint vectors
        hint_vectors = random.binomial(shape=ops.shape(x_disc),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=self._seed_gen)
        hint_vectors *= mask
        # Combine random vectors with original data
        x_disc = x_disc * mask + (1 - mask) * z
        x_generated = self.generator(ops.concatenate([x_disc, mask], axis=1))
        # Combine generated samples with original data
        x_hat_disc = (x_generated * (1 - mask)) + (x_disc * mask)
        with tf.GradientTape() as tape:
            mask_pred = self.discriminator(
                ops.concatenate([x_hat_disc, hint_vectors], axis=1))
            d_loss = self.compute_discriminator_loss(mask, mask_pred)
        gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply(gradients,
                                           self.discriminator.trainable_weights)

        # =====================
        # Train the generator
        # =====================
        mask = 1. - ops.cast(ops.isnan(x_gen), dtype=floatx())
        x_gen = ops.where(ops.isnan(x_gen), x1=0., x2=x_gen)
        z = random.uniform(shape=ops.shape(x_gen), minval=0., maxval=0.01,
                           seed=self._seed_gen)
        hint_vectors = random.binomial(shape=ops.shape(x_gen),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=self._seed_gen)
        hint_vectors *= mask
        x_gen = x_gen * mask + (1 - mask) * z

        with tf.GradientTape() as tape:
            x_generated = self.generator(
                ops.concatenate([x_gen, mask], axis=1))
            # Combine generated samples with original/observed data
            x_hat = (x_generated * (1 - mask)) + (x_gen * mask)
            classifier_pred = self.classifier(x_hat)
            mask_pred = self.discriminator(
                ops.concatenate([x_hat, hint_vectors], axis=1))
            g_loss = self.compute_generator_loss(
                x=x_gen,
                x_generated=x_generated,
                mask=mask,
                mask_pred=mask_pred,
                classifier_pred=classifier_pred,
                alpha=self.alpha,
                beta=self.beta,
            )
        gradients = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply(gradients,
                                       self.generator.trainable_weights)

        # Update custom tracking metrics
        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)

        logs = {m.name: m.result() for m in self.metrics}
        return logs
