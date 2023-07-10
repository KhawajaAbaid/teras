"""
Here we implement regularization layers listed below:

1. MixUp
2. CutMix
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


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
        new_config = {'alpha': self.alpha,
                      }
        config.update(new_config)
        return config


class CutMix(keras.layers.Layer):
    """
    CutMix is a regularization layer proposed by Sangdoo Yun et al.
    in the paper,
    CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features

    It was originally proposed for image data but here it has been
    adapted for Tabular data.

    Args:
        probs: `float`, default 0.3
            CutMix probability which is used in generation of mask that is used
            to mix samples together.
    """
    def __init__(self,
                 probs: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.probs = probs
        self.mask_generator = tfp.distributions.Binomial(total_count=1,
                                                         probs=self.probs)

    def call(self, inputs):
        # Generate mask for CutMix mixing
        mask_cutmix = self.mask_generator.sample(sample_shape=tf.shape(inputs))

        # For each data sample select a partner to mix it with at random.
        # To efficiently achieve this, we can just shuffle the data
        random_partners = tf.random.shuffle(inputs)

        # Apply cutmix formula
        inputs_cutmixed = (inputs * mask_cutmix) + (random_partners * (1 - mask_cutmix))
        return inputs_cutmixed

    def get_config(self):
        config = super().get_config()
        new_config = {'probs': self.probs,
                      }
        config.update(new_config)
        return config
