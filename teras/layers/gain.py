from tensorflow import keras
from tensorflow.keras import layers


class GeneratorBlock(layers.Layer):
    def __init__(self,
                 units,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units,
                                  activation="relu",
                                  kernel_initializer="glorot_normal")

    def call(self, inputs):
        return self.dense(inputs)


class DiscriminatorBlock(layers.Layer):
    def __init__(self,
                 units,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units,
                                  activation="relu",
                                  kernel_initializer="glorot_normal")

    def call(self, inputs):
        return self.dense(inputs)
