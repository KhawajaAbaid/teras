from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.gain")
class GAINDiscriminatorBlock(keras.layers.Layer):
    """
    GAINDiscriminatorBlock that is used in construction of hidden block
    inside the ``GAINDiscriminator`` model of ``GAIN`` architecture.

    Args:
        units: ``int``,
            Hidden dimensionality of the discriminator block

        activation: default "relu",
            Activation function to use in the discriminator block

        kernel_initializer: default "glorot_normal",
            Kernel initializer for the discriminator block
    """
    def __init__(self,
                 units: int,
                 activation="relu",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.dense = keras.layers.Dense(units,
                                        activation=self.activation,
                                        kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        return self.dense(inputs)

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'activation': self.activation,
                      'kernel_initializer': self.kernel_initializer,
                      }

        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        units = config.pop("units")
        return cls(units, **config)