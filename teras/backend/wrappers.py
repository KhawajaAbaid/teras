# This module contains wrapper functions around some Keras 3 functions
# to make them more easy and intuitive to use.
import keras
from keras import ops, random


def flatten(x: keras.KerasTensor):
    """
    Flattens the given tensor/array.

    Args:
        x: ``keras.Tensor``,
            Tensor array to flatten.
    """
    return ops.reshape(x, ops.prod(ops.shape(x)))
