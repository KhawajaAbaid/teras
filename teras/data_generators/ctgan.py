import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from typing import List, Union, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class ResidualBlock(layers.Layer):
    """
    Residual Block as used by the authors of CTGAN proposed
    in the paper Modeling Tabular data using Conditional GAN.

    outputs = Concat([ReLU(BatchNorm(Dense(inputs))), inputs])

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        units: Hidden dimensionality of the dense layer
    """
    def __int__(self,
                units: int,
                **kwargs):
        super().__int__(**kwargs)
        self.units = units
        self.dense = layers.Dense(self.units)
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.concat = layers.Concatenate(axis=1)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        out = self.concat([x, inputs])
        return out


class Generator(layers.Layer):
    """
    Generator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        embedding_dim: Size of the random sample passed to the Generator.
            Defaults to 128.
        generator_dim: a list or tuple of integers. For each value, a Residual block
            of that dimensionality is added to the generator.
            Defaults to [256, 256].
    """
    def __init__(self,
                 embedding_dim: int = 128,
                 generator_dim: LIST_OR_TUPLE = [256, 256],
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        assert isinstance(generator_dim, list) or isinstance(generator_dim, tuple),\
            ("generator_dim must be a list or tuple of integers which determines the number of Residual blocks "
            "and the dimensionality of the hidden layer in those blocks.")
        self.generator_dim = generator_dim
        self.generator = models.Sequential()

        for dim in generator_dim:
            self.generator.add(ResidualBlock(dim))

    def build(self, input_shape):
        dense_out = layers.Dense(input_shape[1])
        self.generator.add(dense_out)

    def call(self, inputs):
        return self.generator(inputs)
