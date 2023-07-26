import tensorflow as tf
from tensorflow import keras
from teras.activations import gumbel_softmax


@keras.saving.register_keras_serializable(package="teras.layers.activation")
class GumbelSoftmax(keras.layers.Layer):
    """
    Implementation of the Gumbel Softmax activation
    proposed by Eric Jang et al. in the paper
    Categorical Reparameterization with Gumbel-Softmax

    Reference(s):
        https://arxiv.org/abs/1611.01144

    Args:
        temperature: ``float``, default 0.2,
            Controls the sharpness or smoothness of the resulting probability distribution.
            A higher temperature value leads to a smoother and more uniform probability distribution.
            Conversely, a lower temperature value makes the distribution concentrated around
            the category with the highest probability.

        hard: ``bool``, default False,
            Whether to return soft probabilities or hard one hot vectors.
    """
    def __init__(self,
                 temperature: float = 0.2,
                 hard: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard

    def call(self, logits: tf.Tensor):
        return gumbel_softmax(logits,
                              temperature=self.temperature,
                              hard=self.hard)

    def get_config(self):
        config = super().get_config()
        new_config = {'temperature': self.temperature,
                      'hard': self.hard}
        config.update(new_config)
        return config
