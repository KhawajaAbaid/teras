from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.ctgan")
class CTGANGeneratorBlock(keras.layers.Layer):
    """
    Residual Block for Generator as used by the authors of CTGAN
    proposed in the paper Modeling Tabular data using Conditional GAN.

    ``outputs = Concat([ReLU(BatchNorm(Dense(inputs))), inputs])``

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        units: ``int``, default 256,
            Dimensionality of the hidden layer
    """
    def __init__(self,
                 units: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = keras.layers.Dense(self.units)
        self.batch_norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.concat = keras.layers.Concatenate(axis=1)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        out = self.concat([x, inputs])
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
