import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


class GeneratorHiddenLayer(keras.layers.Dense):
    def __init__(self,
                 units,
                 activation="relu",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(units=units,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         **kwargs)


class GeneratorOutputLayer(keras.layers.Dense):
    def __init__(self,
                 units,
                 activation="sigmoid",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(units=units,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         **kwargs)