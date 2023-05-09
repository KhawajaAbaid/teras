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
        embedding_dim: Dimensionality of numerical embeddings
        n_features: Number of features
        initialization: Initialization strategy.
        sigma: Used for coefficients initialization
    """
    def __init__(self,
                 embedding_dim: int,
                 n_features: int,
                 initialization: PERIOD_INITIALIZATIONS = 'normal',
                 sigma: float = None,
                 **kwargs):
        super().__init__(**kwargs)
        assert initialization.lower() in ['normal', 'log-linear'], ("Invalid value for initialization."
                                                                    " Must be one of ['log-linear', 'normal']")
        self.embedding_dim = embedding_dim
        self.n_features = n_features
        self.initialization = initialization.lower()
        self.sigma = sigma

        # The official implementation uses another variable n, that is half of the embedding dim
        self.n = self.embedding_dim // 2

    def build(self, input_shape):
        if self.initialization == 'log-linear':
            self.coefficients = self.sigma ** (tf.range(self.n) / self.n)
            self.coefficients = tf.repeat(self.coefficients[None],
                                          repeats=self.n_features,
                                          axis=1)
        else:
            # initialization must be normal
            self.coefficients = tf.random.normal(shape=(self.n_features, self.n),
                                                 mean=0.,
                                                 stddev=self.sigma)

        self.coefficients = tf.Variable(self.coefficients)

    @staticmethod
    def cos_sin(x):
        return tf.concat([tf.cos(x), tf.sin(x)], -1)

    def call(self, inputs):
        assert inputs.ndim == 2
        pi = tf.constant(math.pi)
        return self.cos_sin(2 * pi * self.coefficients[None] * inputs[..., None])
