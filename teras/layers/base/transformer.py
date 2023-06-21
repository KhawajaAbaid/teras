import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.activations import GEGLU
from teras.utils import get_normalization_layer
from typing import List, Tuple, Union


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
LAYER_OR_STR = Union[keras.layers.Layer, str]


class FeedForward(layers.Layer):
    """
    FeedForward layer that is used in the Transformer layer of the Encoder block
    of Transformer architecture.
    It is shared by all transformer based architectures for tabular data.
    The first layer expands the inputs to `embedding dimensions` times `multiplier`
    and the second layer projects them back to the embedding dimensions.
    Usually, since the feed forward layer follows the MultiHeadAttention layer
    whose outputs are of `embedding dimensions` so it essentially takes
    MultiHeadAttention layer's outputs as inputs, expands them to input dimensions
    times the multiplier and then projects them back to the input dimensions.

    Args:
        embedding_dim: `int`, default 16, Embedding dimensions used in the given
            architecture to usually embedded the numerical features but at times
            to also embedd the numerical features like in SAINT architecture.
        multiplier: `int`, default 4, Multiplier that is multipled with embedding dims
            and the resultant value is used as hidden dimensions.
        dropout: Dropout rate to use in the dropout layer that is applied after
            the hidden layer.
        activation: Activation function to use in the hidden layer.
            By default, GEGLU activation is used, which isn't offered by Keras by is offered by Teras.
            You can import it using
            `from teras.layers import GEGLU()`
    """
    def __init__(self,
                 embedding_dim: int = 16,
                 multiplier: int = 4,
                 dropout: float = 0.,
                 activation="geglu",
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.multiplier = multiplier
        self.dropout = dropout
        self.activation = GEGLU() if activation.lower() == "geglu" else activation
        self.dense_1 = layers.Dense(self.embedding_dim * self.multiplier,
                                    activation=self.activation)
        self.dropout = layers.Dropout(self.dropout)
        self.dense_2 = layers.Dense(self.embedding_dim)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


class Transformer(layers.Layer):
    """
    Transformer layer that is used as a building block for
    constructing the Encoder block of transformer based architectures.
    It applies the MultiHeadAttention followed by the FeedForward layer
    and also employs residual connections.

    Args:
        num_attention_heads: `int`, default 8, Number of heads to use
            in the MultiHeadAttention layer.
        embedding_dim: `int`, default 32, emebdding dimensions being
            used in the overall architecture.
            These serve as the `key dimensions` in the
            `MultiHeadAttention` layer and are multiplied by the
            `mutliplier` in the `FeedForward` layer to compute the
            hidden dimensions to project the inputs into.
        attention_dropout: `float`, default 0., Dropout rate for
            the MultiHeadAttention layer.
        feedforward_dropout: `float`, default 0., Dropout rate to use
            in for the dropout layer in the FeedForward layer.
        norm_epsilon: `float`, default 1e-6, Normalization value to
            use for LayerNormalization layer.
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 feedforward_multiplier: int = 4,
                 norm_epsilon: float = 1e-6,
                 **kwagrs):
        super().__init__(**kwagrs)
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon

        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.embedding_dim,
            dropout=self.attention_dropout
        )
        self.skip_1 = layers.Add()
        self.layer_norm_1 = layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.feed_forward = FeedForward(self.embedding_dim,
                                        multiplier=self.feedforward_multiplier,
                                        dropout=self.feedforward_dropout)
        self.skip_2 = layers.Add()
        self.layer_norm_2 = layers.LayerNormalization(epsilon=self.norm_epsilon)

    def call(self, inputs):
        attention_out = self.multi_head_attention(inputs, inputs)
        x = self.skip_1([attention_out, inputs])
        x = self.layer_norm_1(x)
        feedforward_out = self.feed_forward(x)
        x = self.skip_2([feedforward_out, x])
        x = self.layer_norm_2(x)
        return x


class Encoder(layers.Layer):
    """
    Encoder for transformer based architectures.
    It is simply a stack of `num_transformer_layers`
    that is applied to the inputs.

    Args:
        num_transformer_layer: `int`, default 6, Number of transformer layers
            to use in the Encoder.
        num_heads: `int`, default 8, Number of heads to use in the
            MultiHeadAttention layer that is used to construct the Transformer layer.
        embedding_dim: `int`, default 32, Embedding dimensions in the
            MultiHeadAttention layer.
        attention_dropout: `float`, default 0., Dropout rate to use in the
            MultiHeadAttention layer.
        feedforward_dropout: `float`, default 0., Dropout rate to use for
            the dropout layer in the FeedForward layer.
        norm_epsilon: `float`, default 1e-6, Value for epsilon parameter
            of the `LayerNormalization` layer that is used to construct the
            `Transformer` layer.
    """
    def __init__(self,
                 num_transformer_layers: int = 6,
                 num_heads: int = 8,
                 embedding_dim: int = 32,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 feedforward_multiplier: int = 4,
                 norm_epsilon: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon

        self.transformer_layers = models.Sequential(name="transformer_layers")
        for i in range(self.num_transformer_layers):
            self.transformer_layers.add(
                Transformer(num_heads=self.num_heads,
                            embedding_dim=self.embedding_dim,
                            attention_dropout=self.attention_dropout,
                            feedforward_dropout=self.feedforward_dropout,
                            feedforward_multiplier=self.feedforward_multiplier,
                            norm_epsilon=self.norm_epsilon,
                            name=f"transformer_layer_{i+1}"))

    def call(self, inputs):
        outputs = self.transformer_layers(inputs)
        return outputs


class RegressionHead(layers.Layer):
    """
    Regression head to use on top of the transformer based
    architectures for regression.

    Args:
        num_outputs: `int`, default 1, Number of regression outputs to predict.
        units_values_hidden: `List[int] | Tuple[int]`, default (64, 32), for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the RegressionHead.
        activation_hidden: default "relu", Activation function to use in hidden dense layers.
        normalization: `Layer | str`, default "batch", Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 units_values_hidden: LIST_OR_TUPLE = (64, 32),
                 activation_hidden="relu",
                 normalization: LAYER_OR_STR = "batch",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.units_values_hidden = units_values_hidden
        self.activation_hidden = activation_hidden
        self.normalization = normalization

        self.hidden_block = None
        if self.units_values_hidden is not None:
            self.hidden_block = keras.models.Sequential(name="inner_head")
            for units in self.units_values_hidden:
                if self.normalization is not None:
                    self.hidden_block.add(get_normalization_layer(self.normalization))
                self.hidden_block.add(layers.Dense(units,
                                                   activation=self.activation_hidden))
        self.dense_out = layers.Dense(self.num_outputs)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        outputs = self.dense_out(x)
        return outputs


class ClassificationHead(layers.Layer):
    """
    Classification head to use on top of the transformer based
    architectures for classification.

    Args:
        num_classes: `int`, default 2, Number of classes to predict.
        units_values_hidden: `List[int] | Tuple[int]`, default (64, 32), for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the ClassificationHead.
        activation_hidden: default "relu", Activation function to use in hidden dense layers.
        activation_out: Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.
        normalization: `Layer | str`, default "batch", Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_classes: int = 2,
                 units_values_hidden: LIST_OR_TUPLE = (64, 32),
                 activation_hidden="relu",
                 activation_out=None,
                 normalization: LAYER_OR_STR = "batch",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.units_values_hidden = units_values_hidden
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = "sigmoid" if self.num_classes == 1 else "softmax"
        self.normalization = normalization

        self.hidden_block = None
        if self.units_values_hidden is not None:
            self.hidden_block = keras.models.Sequential(name="inner_head")
            for units in self.units_values_hidden:
                if self.normalization is not None:
                    self.hidden_block.add(get_normalization_layer(self.normalization))
                self.hidden_block.add(layers.Dense(units,
                                                   activation=self.activation_hidden))
        self.dense_out = layers.Dense(self.num_classes,
                                 activation=self.activation_out)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        outputs = self.dense_out(x)
        return outputs