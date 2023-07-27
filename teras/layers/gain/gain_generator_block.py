from tensorflow import keras
from teras.utils.types import ActivationType, InitializerType


@keras.saving.register_keras_serializable(package="teras.layers.gain")
class GAINGeneratorBlock(keras.layers.Layer):
    """
    ``GAINGeneratorBlock`` that is used in construction of hidden block
    inside the ``GAINGenerator`` model of ``GAIN`` architecture.

    Args:
        units: ``int``,
            Hidden dimensionality of the generator block

        activation: ``str``, ``keras.initializer.Initializer``, default "relu",
            Activation function to use in the generator block

        kernel_initializer: ``str``, ``keras.initializer.Initializer``, default "glorot_normal",
            Kernel initializer for the generator block
    """
    def __init__(self,
                 units: int,
                 activation: ActivationType = "relu",
                 kernel_initializer: InitializerType = "glorot_normal",
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

        activation_serialized = self.activation
        if not isinstance(self.activation, str):
            activation_serialized = keras.layers.deserialize(self.activation)

        kernel_initializer_serialized = self.kernel_initializer
        if not isinstance(self.activation, str):
            kernel_initializer_serialized = keras.layers.deserialize(self.kernel_initializer)

        config.update({'units': self.units,
                       'activation': activation_serialized,
                       'kernel_initializer': kernel_initializer_serialized,
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        units = config.pop("units")
        return cls(units, **config)
