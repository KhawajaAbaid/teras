"""
This module contains general purpose common layers that can't be
classified into some category or attributed to a base architecture
like transformers.
"""
from tensorflow import keras
from tensorflow.keras import layers


@keras.saving.register_keras_serializable(package="teras.layerflow.layers")
class HiLOL(layers.Layer):
    """
    HiLOL, short for, Hidden Layer Output Layer, is a simple
    layer that is — as the name implies — made up of a
    Hidden Layer and an Output Layer.

    This serves as base to many layers that are specific to
    various model architectures but are made up of this very
    simple architecture,
    for instance, the SAINT's ``ProjectionHead`` layer,
    and almost all LayerFlow versions of Head layers

    This helps us save a lot of code duplication, such as
    implementation of ``call()``, ``get_config()``, and
    ``from_config()`` methods.
    """
    def __init__(self,
                 hidden_block: layers.Layer,
                 output_layer: layers.Layer,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_block = hidden_block
        self.output_layer = output_layer

    def call(self, inputs):
        x = self.hidden_block(inputs)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_block': keras.layers.serialize(self.hidden_block),
                       'output_layer': keras.layers.serialize(self.output_layer)
                       })
        return config

    @classmethod
    def from_config(cls, config):
        hidden_block = keras.layers.deserialize(config.pop("hidden_block"))
        output_layer = keras.layers.deserialize(config.pop("output_layer"))
        return cls(hidden_block, output_layer, **config)