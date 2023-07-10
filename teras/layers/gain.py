from tensorflow.keras import layers


class GeneratorBlock(layers.Layer):
    """
    GeneratorBlock that is used in construction of hidden block
    inside the Generator model of GAIN architecture.

    Args:
        units: `int`,
            Hidden dimensionality of the generator block
        activation: default "relu",
            Activation function to use in the generator block
        kernel_initializer: default "glorot_normal",
            Kernel initializer for the generator block
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
        self.dense = layers.Dense(units,
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


class DiscriminatorBlock(layers.Layer):
    """
    DiscriminatorBlock that is used in construction of hidden block
    inside the Discriminator model of GAIN architecture.

    Args:
        units: `int`,
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
        self.dense = layers.Dense(units,
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
