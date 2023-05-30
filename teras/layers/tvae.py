import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import initializers
from typing import Union, List, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class Encoder(layers.Layer):
    """
    Encoder for the TVAE model as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        latent_dim: Dimensionality of the learned latent space
            Defaults to 128
        compress_dims: A list or tuple of integers. For each value in the sequence,
            a (dense) compression layer is added.
            Defaults to (128, 128)
        activation: activation type to use in the (dense) compression layer.
            Defaults to 'relu'
    """
    def __init__(self,
                 latent_dim: int = 128,
                 compress_dims: LIST_OR_TUPLE = (128, 128),
                 activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.compress_dims = compress_dims
        self.activation = activation

        self.compression_block = models.Sequential(name="compression_block")
        for i, dim in enumerate(self.compress_dims, start=1):
            self.compression_block.add(layers.Dense(units=dim,
                                                          activation=self.activation,
                                                          name=f"compression_layer_{i}"))
        self.dense_mean = layers.Dense(units=self.latent_dim,
                                             name="mean")
        self.dense_log_var = layers.Dense(units=self.latent_dim,
                                               name="log_var")

    def call(self, inputs):
        h = self.compression_block(inputs)
        mean = self.dense_mean(h)
        log_var = self.dense_log_var(h)
        std = tf.exp(0.5 * log_var)
        return mean, std, log_var


class Decoder(layers.Layer):
    """
    Encoder for the TVAE model as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: Dimensionality of the data
        compress_dims: A list or tuple of integers. For each value in the sequence,
            a (dense) decompression layer is added.
            Defaults to (128, 128)
        activation: activation type to use in the (dense) decompression layer.
            Defaults to 'relu'
    """
    def __init__(self,
                 data_dim: int = None,
                 decompress_dims: LIST_OR_TUPLE = None,
                 activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.decompress_dims = decompress_dims
        self.activation = activation

        self.decompress_block = models.Sequential(name="decompress_block")
        for i, dim in enumerate(self.decompress_dims, start=1):
            self.decompress_block.add(layers.Dense(units=dim,
                                                         activation=self.activation,
                                                         name=f"decompress_layer_{i}"
                                                         )
                                      )

        self.decompress_block.add(layers.Dense(units=self.data_dim,
                                                     name="projection_to_data_dim"))
        self.sigmas = tf.Variable(initial_value=initializers.ones()(shape=(self.data_dim,)) * 0.1,
                                 trainable=True,
                                 name="sigmas")

    def call(self, inputs):
        x_generated = self.decompress_block(inputs)
        return x_generated, self.sigmas
