from tensorflow import keras
from teras.layerflow.layers.common.common import HiLOL
from teras.utils import (serialize_layers_collection,
                         deserialize_layers_collection)
from typing import List

LIST_OF_LAYERS = List[keras.layers.Layer]


class FeedForward(HiLOL):
    """
    FeedForward layer with LayerFlow design.
    It is used in the Transformer layer of the ``Encoder`` layer
    of Transformer architecture.
    It is shared by all transformer based architectures for tabular data.

    Args:
        hidden_block: ``keras.layers.Layer``,
            A dense layer or any custom layer that can serve as the hidden
            block for the ``FeedForward`` layer.

        output_layer: ``keras.layers.Layer``,
            A dense layer or any custom layer that projects data back to
            the input/embedding dimensions. Since the ``FeedForward`` receives
            feature embeddings as inputs, it after apply hidden block,
            projects them back to the embedding dimensions.
    """
    def __init__(self,
                 hidden_block: keras.layers.Layer,
                 output_layer: keras.layers.Layer,
                 **kwargs):
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)


class Transformer(keras.layers.Layer):
    """
    Transformer layer with LayerFlow design.
    It is used to made up the ``Encoder`` layer of the given
    Transformer-based architecture.

    Args:
        multi_head_attention: ``keras.layers.Layer``,
            An instance of ``MultiHeadAttention`` layer or any layer
            that can be sued in place of ``MultiHeadAttention`` layer.

        feed_forward: ``keras.layers.Layer``,
            An instance of ``FeedForward`` layer or any layer that
            can be used in its place.
            You can import the ``FeedForward`` layer as follows,
                >>> from teras.layerflow.layers import FeedForward
    """
    def __init__(self,
                 multi_head_attention: keras.layers.Layer,
                 feed_forward: keras.layers.Layer,
                 **kwargs):
        super().__init__(**kwargs)
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward

    def get_config(self):
        config = super().get_config()
        new_config = {'multi_head_attention': keras.layers.serialize(self.multi_head_attention),
                      'feed_forward': keras.layers.serialize(self.feed_forward),
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        multi_head_attention = keras.layers.deserialize(config.pop("multi_head_attention"))
        feed_forward = keras.layers.deserialize(config.pop("feed_forward"))
        transformer = keras.layers.deserialize(config.pop("transformer"))
        return cls(multi_head_attention=multi_head_attention,
                   feed_forward=feed_forward,
                   transformer=transformer,
                   **config)


class Encoder(keras.layers.Layer):
    """
    Encoder layer with LayerFlow design.
    It is made up of several ``Transformer`` layers and encodes the
    input embeddings to useful representations that can then be used
    for the task at hand, such as classificaiton or regression.

    Args:
        transformer_layers: ``List[keras.layers.Layer]``,
            A list of ``Transformer`` layers that make up the encoder
            layer.
            You can import the ``Transformer`` layer as follows,
                >>> from teras.layerflow.layers import Transformer
    """
    def __init__(self,
                 transformer_layers: LIST_OF_LAYERS,
                 **kwargs):
        super().__init__(**kwargs)
        if transformer_layers is not None:
            self.transformer_layers = keras.models.Sequential(
                transformer_layers,
                name="transformer_layers"
            )

    def get_config(self):
        config = super().get_config()
        config.update({'transformer_layers': serialize_layers_collection(self.transformer_layers)})
        return config

    @classmethod
    def from_config(cls, config):
        transformer_layers = deserialize_layers_collection(config.pop("transformer_layers"))
        return cls(transformer_layers, **config)
