from tensorflow import keras
from teras.utils.types import ActivationType


@keras.saving.register_keras_serializable(package="teras.layers.vime")
class VimeMaskEstimator(keras.layers.Layer):
    """
    VimeMaskEstimator layer based on the
    ``VIME`` architecture proposed by Jinsung Yoon et a.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        units: ``int``, default 32,
            Dimensionality of Mask Estimator layer

        activation: ``str`` or ``callable`` or ``keras.layers.Layer``, default "sigmoid",
            Activation function to use.
    """
    def __init__(self,
                 units: int = 32,
                 activation: ActivationType = "sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.estimator = keras.layers.Dense(self.units,
                                            activation=self.activation)

    def call(self, inputs):
        return self.estimator(inputs)

    def get_config(self):
        config = super().get_config()
        activation_serialized = self.activation
        if not isinstance(self.activation, str):
            activation_serialized = keras.layers.serialize(self.activation)
        config.update({'units': self.units,
                       'activation': activation_serialized,
                       }
                      )
        return config
