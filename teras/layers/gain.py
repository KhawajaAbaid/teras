import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


class GeneratorHiddenLayer(layers.Layer):
    def __init__(self,
                 units,
                 activation="relu",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units=units,
                                  activation=activation,
                                  kernel_initializer=kernel_initializer)

    def call(self, inputs):
        return self.dense(inputs)


class GeneratorOutputLayer(layers.Layer):
    def __init__(self,
                 units,
                 activation="sigmoid",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units=units,
                                  activation=activation,
                                  kernel_initializer=kernel_initializer)

    def call(self, inputs):
        return self.dense(inputs)


class DiscriminatorHiddenLayer(layers.Layer):
    def __init__(self,
                 units,
                 activation="relu",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units=units,
                                  activation=activation,
                                  kernel_initializer=kernel_initializer)

    def call(self, inputs):
        return self.dense(inputs)


class DiscriminatorOutputLayer(layers.Layer):
    def __init__(self,
                 units,
                 activation="sigmoid",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(units=units,
                                  activation=activation,
                                  kernel_initializer=kernel_initializer)

    def call(self, inputs):
        return self.dense(inputs)