import tensorflow as tf
import keras
from keras import random, ops
from teras._src.backend.common.models.gans.ctgan.ctgan import BaseCTGAN


class CTGAN(BaseCTGAN):
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 metadata: dict,
                 latent_dim: int = 128,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         metadata=metadata,
                         latent_dim=latent_dim,
                         seed=seed,
                         **kwargs)

    def train_step(self, data):
        actual_data, dummy_vals = data
        del dummy_vals
        x, cond_vectors_real, cond_vectors, mask = actual_data
        batch_size = ops.shape(x)[0]

        # =========================
        # Discriminator train step
        # =========================
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)
        x_generated = self.generator(input_gen)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        x = ops.concatenate([x, cond_vectors_real],
                            axis=1)

        with tf.GradientTape(persistent=True) as tape:
            y_pred_generated = self.discriminator(x_generated)
            y_pred_real = self.discriminator(x)
            grad_pen = self.discriminator.gradient_penalty(x,
                                                           x_generated)
            loss_disc = self.compute_disciriminator_loss(
                y_pred_real,
                y_pred_generated)
        gradients_pen = tape.gradient(grad_pen,
                                      self.discriminator.trainable_variables)
        gradients_loss = tape.gradient(loss_disc,
                                       self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply(
            gradients_pen,
            self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply(
            gradients_loss,
            self.discriminator.trainable_variables)

        # =====================
        # Generator train step
        # =====================
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(cond_vectors)
            tape.watch(mask)
            x_generated = self.generator(input_gen)
            x_generated = ops.concatenate(
                [x_generated, cond_vectors], axis=1)
            y_pred_generated = self.discriminator(x_generated)
            loss_gen = self.compute_generator_loss(
                x_generated, y_pred_generated,
                cond_vectors=cond_vectors, mask=mask,
                metadata=self.metadata)
        gradients = tape.gradient(loss_gen, self.generator.trainable_variables)
        self.generator_optimizer.apply(gradients,
                                       self.generator.trainable_variables)

        self.generator_loss_tracker.update_state(loss_gen)
        self.discriminator_loss_tracker.update_state(loss_disc)

        results = {m.name: m.result() for m in self.metrics}
        return results
