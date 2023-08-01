import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


@keras.saving.register_keras_serializable(package="teras.layers.regularization")
class MixUp(keras.layers.Layer):
    """
    MixUp is a regularization layer proposed by Hongyi Zhang et al.
    in the paper,
    mixup: BEYOND EMPIRICAL RISK MINIMIZATION

    It was originally proposed for image data but here it has been
    adapted for Tabular data.

    Reference(s):
        https://arxiv.org/abs/1710.09412

    Args:
        alpha: `float`, default 1.0,
            Parameter for the Beta distribution to sample `lambda_`
            from which is used to interpolate samples.
    """
    def __init__(self,
                 alpha: float = 1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_generator = tfp.distributions.Beta(self.alpha, self.alpha)

    def call(self, inputs):
        # Sample lambda_
        lambda_ = tf.squeeze(self.lambda_generator.sample(1))
        # For each data sample select a partner to mix it with at random.
        # To efficiently achieve this, we can just shuffle the data
        random_partners = tf.random.shuffle(inputs)

        # SIDE NOTE:
        # We could make the shuffling more memory efficient by just shuffling the indices
        # but the problem with that approach is we will have to check and handle the
        # dictionary type data differently -- making it lot more complex and perhaps
        # worse in terms of efficiency

        inputs_mixedup = (lambda_ * inputs) + (1 - lambda_) * random_partners
        return inputs_mixedup

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha,
                       })
        return config
