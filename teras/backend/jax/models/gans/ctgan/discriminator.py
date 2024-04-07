import jax
from keras import random, ops
from teras.backend.common.models.gans.ctgan.discriminator import BaseCTGANDiscriminator
from teras.utils.types import IntegerSequence


class CTGANDiscriminator(BaseCTGANDiscriminator):
    def __init__(self,
                 hidden_dims: IntegerSequence = (256, 256),
                 packing_degree: int = 8,
                 gradient_penalty_lambda: float = 10.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(hidden_dims=hidden_dims,
                         packing_degree=packing_degree,
                         gradient_penalty_lambda=gradient_penalty_lambda,
                         seed=seed,
                         **kwargs)

    def compute_y_interpolated(self,
                               trainable_variables,
                               non_trainable_variables,
                               interpolated_samples
                               ):
        y_interpolated = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            interpolated_samples,
            training=True
        )
        return y_interpolated

    def gradient_penalty(self,
                         trainable_variables,
                         non_trainable_variables,
                         real_samples,
                         generated_samples):
        batch_size = ops.shape(real_samples)[0]
        dim = ops.shape(real_samples)[1]

        seed = non_trainable_variables[0]

        alpha = random.uniform(
            shape=(batch_size // self.packing_degree, 1, 1),
            seed=seed,
        )
        alpha = ops.reshape(
            ops.tile(alpha, [1, self.packing_degree, dim]),
            (-1, dim))
        interpolated_samples = ((alpha * real_samples)
                                + ((1 - alpha) * generated_samples))

        grad_fn = jax.value_and_grad(self.compute_y_interpolated)
        y_interpolated, gradients = grad_fn(trainable_variables,
                                            non_trainable_variables,
                                            interpolated_samples)
        gradients = ops.reshape(gradients,
                                shape=(-1, self.packing_degree * dim))
        # Calculating gradient penalty
        gradients_norm = ops.norm(gradients)

        gradient_penalty = (
                ops.mean(ops.square(gradients_norm - 1.0)) *
                self.gradient_penalty_lambda
        )
        seed = jax.random.split(seed, 1)[0]
        non_trainable_variables[0] = seed
        return gradient_penalty, non_trainable_variables
