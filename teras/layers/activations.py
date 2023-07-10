import tensorflow as tf
from tensorflow.keras import layers

class GLU(layers.Layer):
    """Generalized linear unit nonlinear activation."""
    def __init__(self,
                units,
                **kwagrs):
        super().__init__(**kwagrs)
        self.units = units
    
    def call(self, inputs):
        return inputs[:, :self.units] * tf.nn.sigmoid(inputs[:, self.units:])

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units}
        config.update(new_config)
        return config


class GEGLU(layers.Layer):
    """GeGLU is an activation function which is a variant of GLU"""
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x, gates = tf.split(inputs,
                            num_or_size_splits=2,
                            axis=-1)
        return x * tf.nn.gelu(gates)


class GumbelSoftmax(layers.Layer):
    """
    Implementation of the Gumbel Softmax activation
    proposed by Eric Jang et al. in the paper
    Categorical Reparameterization with Gumbel-Softmax

    Reference(s):
        https://arxiv.org/abs/1611.01144

    Args:
        temperature: Controls the sharpness or smoothness of the resulting probability distribution.
            A higher temperature value leads to a smoother and more uniform probability distribution.
            Conversely, a lower temperature value makes the distribution concentrated around
            the category with the highest probability. Defaults to 0.2
        hard: Whether to return soft probabilities or hard one hot vectors. Defaults to False.
    """
    def __init__(self,
                 temperature=0.2,
                 hard: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard

    def call(self, logits):
        u = tf.random.uniform(tf.shape(logits),
                                    minval=0,
                                    maxval=1)
        gumbels = -tf.math.log(-tf.math.log(u))
        perturbed_logits = (logits + gumbels) / self.temperature
        probabilities = tf.nn.softmax(perturbed_logits)
        if self.hard:
            one_hot_labels = tf.one_hot(tf.argmax(probabilities, axis=-1), tf.shape(logits)[-1])
            return one_hot_labels
        return probabilities

    def get_config(self):
        config = super().get_config()
        new_config = {'temperature': self.temperature,
                      'hard': self.hard}
        config.update(new_config)
        return config
