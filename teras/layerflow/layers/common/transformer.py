from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.common.transformer import (Transformer as BaseTransformer,
                                             Encoder as BaseEncoder,
                                             RegressionHead as BaseRegressionHead,
                                             ClassificationHead as BaseClassificationHead)
from typing import List

LIST_OF_LAYERS = List[keras.layers.Layer]


class Transformer(BaseTransformer):
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


class Encoder(BaseEncoder):
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


# TODO: Append these Base.* layer names with an underscore,
#       so it becomes cleaner when the user tries to import layers

class ClassificationHead(BaseClassificationHead):
    """
    ClassificationHead with LayerFlow design.
    It's purposed to be used on top of the transformer based
    architectures for classification.

    Args:
        hidden_block: `layers.Layer | models.Model`,
            An instance of anything that can serve as the hidden block in the
            classification head.
            It can be as simple as a single dense layer, or a custom layer that
            uses a bunch of other dense and other fancy layers,
            or may as well be a keras model -- as long as it satisfies the input
            output constraints. (All thanks to Keras and Francois Chollet :D)
        output_layer: `layers.Layer`,
            An instance of keras layer (Dense or a custom layer), with relevant
            activation function for classification relevant to the task at hand.
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


class RegressionHead(BaseRegressionHead):
    """
    RegressionHead with LayerFlow design.
    It's purposed to be used on top of the transformer based
    architectures for regression.

    Args:
        hidden_block: `layers.Layer | models.Model`,
            An instance of anything that can serve as the hidden block in the
            regression head.
            It can be as simple as a single dense layer, or a custom layer that
            uses a bunch of other dense and other fancy layers,
            or may as well be a keras model -- as long as it satisfies the input
            output constraints. (All thanks to Keras and Francois Chollet :D)
        output_layer: `layers.Layer`,
            An instance of keras layer (Dense or a custom layer),
            for regression outputs relevant to the task at hand.
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
