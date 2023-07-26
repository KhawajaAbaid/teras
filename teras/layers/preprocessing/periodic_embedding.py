import tensorflow as tf
from tensorflow import keras
from typing import Literal
import math

PERIOD_INITIALIZATIONS = Literal["log-linear", "normal"]


class PeriodicEmbedding(keras.layers.Layer):
    """
    PeriodicEmbedding layer for numerical features
    as proposed Yury Gorishniy et al. in the paper
    On Embeddings for Numerical Features in Tabular Deep Learning, 2022.

    Reference(s):
        https://arxiv.org/abs/2203.05556

    Args:
        data_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        embedding_dim: ``int``, default 32,
            Dimensionality of numerical embeddings

        initialization: default "normal",
            Initialization strategy.

        sigma: ``float``, default 0.01,
            Used for coefficients initialization
    """
    def __init__(self,
                 data_dim: int,
                 embedding_dim: int = 32,
                 initialization: PERIOD_INITIALIZATIONS = "normal",
                 sigma: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        if initialization.lower() not in ["normal", "log-linear"]:
            raise ValueError("Invalid value for initialization. Must be one of ['log-linear', 'normal']\n"
                             f"Received: {initialization}")
        self.embedding_dim = embedding_dim
        self.data_dim = data_dim
        self.initialization = initialization.lower()
        self.sigma = sigma

        # The official implementation uses another variable n, that is half of the embedding dim
        self.n = self.embedding_dim // 2
        self.coefficients = None

    def build(self, input_shape):
        if self.initialization == 'log-linear':
            self.coefficients = self.sigma ** (tf.range(self.n) / self.n)
            self.coefficients = tf.repeat(self.coefficients[None],
                                          repeats=self.data_dim,
                                          axis=1)
        else:
            # initialization must be normal
            self.coefficients = tf.random.normal(shape=(self.data_dim, self.n),
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
        config.update({'data_dim': self.data_dim,
                       'embedding_dim': self.embedding_dim,
                       'initialization': self.initialization,
                       'sigma': self.sigma,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim,
                   **config)
