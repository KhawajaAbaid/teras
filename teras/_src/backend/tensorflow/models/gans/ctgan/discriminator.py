import tensorflow as tf
from keras import random, ops
from teras._src.backend.common.models.gans.ctgan.discriminator import BaseCTGANDiscriminator
from teras._src.typing import IntegerSequence


class CTGANDiscriminator(BaseCTGANDiscriminator):
    def __init__(self,
                 hidden_dims: IntegerSequence = (256, 256),
                 packing_degree: int = 8,
                 gradient_penalty_lambda: float = 10.,
                 **kwargs):
        super().__init__(hidden_dims=hidden_dims,
                         packing_degree=packing_degree,
                         gradient_penalty_lambda=gradient_penalty_lambda,
                         **kwargs)

    def gradient_penalty(self,
                         real_samples,
                         generated_samples):
        batch_size = ops.shape(real_samples)[0]
        dim = ops.shape(real_samples)[1]

        alpha = random.uniform(
            shape=(batch_size // self.packing_degree, 1, 1),
            seed=self.seed_gen,
        )
        alpha = ops.reshape(ops.tile(alpha, [1, self.packing_degree, dim]),
                            (-1, dim))
        interpolated_samples = ((alpha * real_samples)
                                + ((1 - alpha) * generated_samples))

        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            y_interpolated = self(interpolated_samples)
        gradients = tape.gradient(y_interpolated, interpolated_samples)
        gradients = ops.reshape(gradients,
                                newshape=(-1, self.packing_degree * dim))
        # Calculating gradient penalty
        gradients_norm = tf.norm(gradients)

        gradient_penalty = (
                ops.mean(ops.square(gradients_norm - 1.0)) *
                self.gradient_penalty_lambda
        )
        return gradient_penalty
