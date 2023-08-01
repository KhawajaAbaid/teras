from tensorflow import keras
from teras.utils import (serialize_layers_collection,
                         deserialize_layers_collection)
from typing import List, Union

LAYERS_COLLECTION = Union[keras.layers.Layer, List[keras.layers.Layer], keras.models.Model]


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class SAINTEncoder(keras.layers.Layer):
    """
    SAINTEncoder layer with LayerFlow desing.
    It is part of the SAINT architecture,
    which is proposed by Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It simply stacks N transformer layers and applies them to the outputs
    of the embedded features.

    It differs from the typical Encoder block only in that the Transformer
    layer is a bit different from the regular Transformer layer used in the
    Transformer based architectures as it uses multi-head inter-sample attention,
    in addition to the regular mutli-head attention for features.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        saint_transformer_layers: ``keras.layers.Layer`` or ``List[Layer]`` or ``models.Model``,
            A list, a keras layer or a keras model made up of ``SAINTTransformer`` layers or
            any other custom layers for that purpose.
            You can import the ``SAINTTransformer`` layer as follows,
                >>> from teras.layerflow.layers import SAINTTransformer
    """
    def __init__(self,
                 saint_transformer_layers: LAYERS_COLLECTION,
                 **kwargs):
        super().__init__(**kwargs)
        self.saint_transformer_layers = saint_transformer_layers

        if isinstance(self.saint_transformer_layers, (list, tuple)):
            self.saint_transformer_layers = keras.models.Sequential(self.saint_transformer_layers,
                                                                    name="saint_transformer_layers")

    def call(self, inputs):
        return self.saint_transformer_layers(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'saint_transformer_layers': serialize_layers_collection(self.saint_transformer_layers)})
        return config

    @classmethod
    def from_config(cls, config):
        saint_transformer_layers = deserialize_layers_collection(config.pop("saint_transformer_layers"))
        return cls(saint_transformer_layers, **config)