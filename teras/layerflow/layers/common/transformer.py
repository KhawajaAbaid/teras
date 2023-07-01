from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.common.transformer import (Transformer as _BaseTransformer,
                                             Encoder as _BaseEncoder)
from typing import List

LIST_OF_LAYERS = List[keras.layers.Layer]


class Transformer(_BaseTransformer):
    """
    Transformer layer with LayerFlow design.
    It supposed to be used when user wants additional flexibility
    to customize the MultiHeadAttention and FeedForward layers,
    that is not offered by the default Teras API.

    Args:
        multi_head_attention: `layers.Layer`,
            An instance of MultiHeadAttention layer or any layer
            that can be sued in place of MultiHeadAttention layer.
        feed_forward: `layers.Layer`,
            An instance of FeedForward layer or any layer that
            can be used in place of FeedForward layer.
    """
    def __init__(self,
                 multi_head_attention: layers.Layer = None,
                 feed_forward: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if multi_head_attention is not None:
            self.multi_head_attention = multi_head_attention

        if feed_forward is not None:
            self.feed_forward = feed_forward


class Encoder(_BaseEncoder):
    """
    Encoder layer with LayerFlow design.
    It supposed to be used when user wants additional flexibility
    to customize the Transformer layers,
    that is not offered by the default Teras API.

    Args:
        transformer_layers: `List[layers.Layer]`,
            A list of Transformer layers that make up the encoder
            layer.
    """
    def __init__(self,
                 transformer_layers: LIST_OF_LAYERS = None,
                 **kwargs):
        super().__init__(**kwargs)
        if transformer_layers is not None:
            self.transformer_layers = models.Sequential(
                transformer_layers,
                name="transformer_layers"
            )


# We don't need the common classification and regression heads for
# the layerflow API, why? because in default API it makes sense to
# have the common heads since they allow parameters values to customize
# the structure and behavior for the given architecture,
# but here, the common heads will have the exact same interface i.e.
# the exact same two parameters as the architecture specific ones
# NOT only does it not make sense to have them but ALSO there's
# no way to customize these for the specific architectures.
# The better solution is to use the architecture specific heads from
# the default api as base for the architecture specific heads in the
# layerflow api

