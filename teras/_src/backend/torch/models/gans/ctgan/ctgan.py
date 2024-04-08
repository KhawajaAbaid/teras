import torch
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
        x, cond_vectors_real, cond_vectors, mask = data
        batch_size = ops.shape(x)[0]

        # =========================
        # Discriminator train step
        # =========================
        self.zero_grad()
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)
        x_generated = self.generator(input_gen, training=False)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        x = ops.concatenate([x, cond_vectors_real],
                            axis=1)

        y_pred_generated = self.discriminator(x_generated)
        y_pred_real = self.discriminator(x)
        grad_pen = self.discriminator.gradient_penalty(x,
                                                       x_generated)
        loss_disc = self.discriminator_loss(y_pred_real, y_pred_generated)
        grad_pen.backward(retain_graph=True)
        grads = [v.value.grad for v in self.discriminator.trainable_variables]
        with torch.no_grad():
            self.discriminator_optimizer.apply(
                grads,
                self.discriminator.trainable_variables)
        loss_disc.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_variables]
        with torch.no_grad():
            self.discriminator_optimizer.apply(
                grads,
                self.discriminator.trainable_variables)

        # =====================
        # Generator train step
        # =====================
        self.zero_grad()
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)
        x_generated = self.generator(input_gen)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        y_pred_generated = self.discriminator(x_generated)
        loss_gen = self.generator_loss(x_generated, y_pred_generated,
                                       cond_vectors=cond_vectors, mask=mask,
                                       metadata=self.metadata)
        loss_gen.backward()
        grads = [v.value.grad for v in self.generator.trainable_variables]
        self.generator_optimizer.apply(grads,
                                       self.generator.trainable_variables)

        self.generator_loss_tracker.update_state(loss_gen)
        self.discriminator_loss_tracker.update_state(loss_disc)

        results = {m.name: m.result() for m in self.metrics}
        return results
