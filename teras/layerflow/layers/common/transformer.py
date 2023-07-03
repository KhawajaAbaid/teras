from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.common.transformer import (Transformer as _BaseTransformer,
                                             Encoder as _BaseEncoder,
                                             FeedForward as _BaseFeedForward)
from typing import List

LIST_OF_LAYERS = List[keras.layers.Layer]


class FeedForward(_BaseFeedForward):
    """
    FeedForward layer with LayerFlow design.
    It is used in the Transformer layer of the Encoder block
    of Transformer architecture.
    It is shared by all transformer based architectures for tabular data.

    Args:
        hidden_block: `layers.Layer`,
            A dense layer or any custom layer that can serve as the hidden
            block for the FeedForward layer.
            If None, a hidden block with default values is used.
        output_layer: `layers.Layer`,
            A dense layer or any custom layer that projects data back to
            the input/embedding dimensions. Since the FeedForward recieves
            feature embeddings as inputs, it after apply hidden block,
            projects them back to the embedding dimensions.
            If None, an output layer with default value is used.
    """
    def __init__(self,
                 hidden_block: layers.Layer = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer


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
            You can import the `FeedForward` layer as follows,
                >>> from teras.layerflow.layers import FeedForward
                or
                >>> from teras.layers import FeedForward
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
            A list of `Transformer` layers that make up the encoder
            layer.
            You can import the `Transformer` layer as follows,
                >>> from teras.layerflow.layers import Transformer
                or
                >>> from teras.layers import Transformer
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

