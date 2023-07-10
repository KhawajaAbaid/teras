from tensorflow.keras import layers
import tensorflow as tf
from typing import Literal
import math

PERIOD_INITIALIZATIONS = Literal['log-linear', 'normal']


class PeriodicEmbedding(layers.Layer):
    """Period embedding layer for numerical features
    as proposed Yury Gorishniy et al. in the paper
    On Embeddings for Numerical Features in Tabular Deep Learning, 2022.

    Reference(s):
        https://arxiv.org/abs/2203.05556

    Args:
        num_features: `int`,
            Number of features in the dataset.
        embedding_dim: `int`, default 32,
            Dimensionality of numerical embeddings
        initialization: default "normal",
            Initialization strategy.
        sigma: `float`, default 0.01
            Used for coefficients initialization
    """
    def __init__(self,
                 num_features: int,
                 embedding_dim: int = 32,
                 initialization: PERIOD_INITIALIZATIONS = 'normal',
                 sigma: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        assert initialization.lower() in ['normal', 'log-linear'], ("Invalid value for initialization."
                                                                    " Must be one of ['log-linear', 'normal']")
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.initialization = initialization.lower()
        self.sigma = sigma

        # The official implementation uses another variable n, that is half of the embedding dim
        self.n = self.embedding_dim // 2

    def build(self, input_shape):
        if self.initialization == 'log-linear':
            self.coefficients = self.sigma ** (tf.range(self.n) / self.n)
            self.coefficients = tf.repeat(self.coefficients[None],
                                          repeats=self.num_features,
                                          axis=1)
        else:
            # initialization must be normal
            self.coefficients = tf.random.normal(shape=(self.num_features, self.n),
                                                 mean=0.,
                                                 stddev=self.sigma)

        self.coefficients = tf.Variable(self.coefficients)

    @staticmethod
    def cos_sin(x):
        return tf.concat([tf.cos(x), tf.sin(x)], -1)

    def call(self, inputs):
        pi = tf.constant(math.pi)
        return self.cos_sin(2 * pi * self.coefficients[None] * inputs[..., None])

    def get_config(self):
        config = super().get_config()
        new_config = {'num_features': self.num_features,
                      'embedding_dim': self.embedding_dim,
                      'initialization': self.initialization,
                      'sigma': self.sigma,
                      }

        config.update(new_config)
        return config
