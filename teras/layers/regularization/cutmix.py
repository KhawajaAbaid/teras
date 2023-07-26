import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


@keras.saving.register_keras_serializable(package="teras.layers.regularization")
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
        config.update({'probs': self.probs,
                       })
        return config
