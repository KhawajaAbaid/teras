from tensorflow import keras
from teras.utils import get_activation
from teras.layerflow.layers.common.transformer import (FeedForward as _FeedForwardLF,
                                                       Transformer as _TransformerLF,
                                                       Encoder as _EncoderLF)
from teras.utils.types import ActivationType


@keras.saving.register_keras_serializable(package="teras.layers.common")
class FeedForward(_FeedForwardLF):
    """
    FeedForward layer that is used in the ``Transformer`` layer of the ``Encoder`` block
    of Transformer architecture.
    It is shared by all transformer based architectures for tabular data.
    The first layer expands the inputs to `embedding dimensions` times `multiplier`
    and the second layer projects them back to the embedding dimensions.
    Usually, since the feed forward layer follows the ``MultiHeadAttention`` layer
    whose outputs are of `embedding dimensions` so it essentially takes
    MultiHeadAttention layer's outputs as inputs, expands them to input dimensions
    times the multiplier and then projects them back to the input dimensions.

    Args:
        embedding_dim: `int`, default 16,
            Embedding dimensions used in the given
            architecture to usually embedded the numerical features but at times
            to also embedd the numerical features like in SAINT architecture.

        multiplier: ``int``, default 4,
            Multiplier that is multipled with embedding dims
            and the resultant value is used as hidden dimensions.

        dropout: ``float``, default 0.,
            Dropout rate to use in the dropout layer that is applied after
            the hidden layer.

        activation: default "geglu",
            Activation function to use in the hidden layer.
            By default, ``geglu`` activation is used.
            You can import it as,
                >>> from teras.activations import geglu
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 multiplier: int = 4,
                 dropout: float = 0.,
                 activation: ActivationType = "geglu",
                 **kwargs):
        hidden_block = keras.models.Sequential(name="feed_forward_hidden_block")
        hidden_block.add(keras.layers.Dense(embedding_dim * multiplier,
                                            activation=get_activation(activation)))
        hidden_block.add(keras.layers.Dropout(dropout))
        output_layer = keras.layers.Dense(embedding_dim)

        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)

        self.embedding_dim = embedding_dim
        self.multiplier = multiplier
        self.dropout = dropout
        self.activation = activation

    def get_config(self):
        activation_serialized = self.activation
        if not isinstance(activation_serialized, str):
            activation_serialized = keras.layers.serialize(activation_serialized)
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'embedding_dim': self.embedding_dim,
                  'multiplier': self.multiplier,
                  'dropout': self.dropout,
                  'activation': activation_serialized}
        return config


@keras.saving.register_keras_serializable(package="teras.layers.common")
class Transformer(_TransformerLF):
    """
    Transformer layer that is used as a building block for
    constructing the ``Encoder`` block of transformer based architectures.
    It applies the ``MultiHeadAttention`` followed by the FeedForward layer
    and also employs residual connections.

    Args:
        num_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadAttention`` layer.

        embedding_dim: ``int``, default 32,
            Embedding dimensions being used in the overall architecture.
            These serve as the `key dimensions` in the
            ``MultiHeadAttention`` layer and are multiplied by the
            ``mutliplier`` in the ``FeedForward`` layer to compute the
            hidden dimensions to project the inputs into.

        attention_dropout: ``float``, default 0.,
            Dropout rate for the ``MultiHeadAttention`` layer.

        feedforward_dropout: ``float``, default 0.,
            Dropout rate to use in for the dropout layer in the ``FeedForward`` layer.

        norm_epsilon: ``float``, default 1e-6,
            Normalization value to use for ``LayerNormalization`` layer.
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 feedforward_multiplier: int = 4,
                 norm_epsilon: float = 1e-6,
                 **kwagrs):
        multi_head_attention = keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=embedding_dim,
            dropout=attention_dropout
        )
        layer_norm_1 = keras.layers.LayerNormalization(epsilon=norm_epsilon)
        feed_forward = FeedForward(embedding_dim,
                                   multiplier=feedforward_multiplier,
                                   dropout=feedforward_dropout)
        layer_norm_2 = keras.layers.LayerNormalization(epsilon=norm_epsilon)
        super().__init__(multi_head_attention=multi_head_attention,
                         feed_forward=feed_forward,
                         layer_norm_1=layer_norm_1,
                         layer_norm_2=layer_norm_2,
                         **kwagrs)
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'embedding_dim': self.embedding_dim,
                  'num_attention_heads': self.num_attention_heads,
                  'attention_dropout': self.attention_dropout,
                  'feedforward_dropout': self.feedforward_dropout,
                  'feedforward_multiplier': self.feedforward_multiplier,
                  'norm_epsilon': self.norm_epsilon}
        return config


@keras.saving.register_keras_serializable(package="teras.layers.common")
class Encoder(_EncoderLF):
    """
    Encoder for transformer based architectures.
    It is simply a stack of `num_transformer_layers`
    that is applied to the inputs.

    Args:
        num_transformer_layer: ``int``, default 6,
            Number of transformer layers to use in the Encoder.

        num_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadAttention`` layer
            that is used to construct the ``Transformer`` layer.

        embedding_dim: ``int``, default 32,
            Embedding dimensions being used in the overall architecture.
            These serve as the `key dimensions` in the
            ``MultiHeadAttention`` layer and are multiplied by the
            ``mutliplier`` in the ``FeedForward`` layer to compute the
            hidden dimensions to project the inputs into.

        attention_dropout: ``float``, default 0.,
            Dropout rate for the ``MultiHeadAttention`` layer.

        feedforward_dropout: ``float``, default 0.,
            Dropout rate to use in for the dropout layer in the ``FeedForward`` layer.

        norm_epsilon: ``float``, default 1e-6,
            Normalization value to use for ``LayerNormalization`` layer.
    """
    def __init__(self,
                 num_transformer_layers: int = 6,
                 num_attention_heads: int = 8,
                 embedding_dim: int = 32,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 feedforward_multiplier: int = 4,
                 norm_epsilon: float = 1e-6,
                 **kwargs):
        transformer_layers = keras.models.Sequential(name="transformer_layers")
        for i in range(num_transformer_layers):
            transformer_layers.add(
                Transformer(num_attention_heads=num_attention_heads,
                            embedding_dim=embedding_dim,
                            attention_dropout=attention_dropout,
                            feedforward_dropout=feedforward_dropout,
                            feedforward_multiplier=feedforward_multiplier,
                            norm_epsilon=norm_epsilon,
                            name=f"transformer_layer_{i+1}"))
        super().__init__(transformer_layers,
                         **kwargs)
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.embedding_dim = embedding_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'num_transformer_layers': self.num_transformer_layers,
                  'num_attention_heads': self.num_attention_heads,
                  'embedding_dim': self.embedding_dim,
                  'attention_dropout': self.attention_dropout,
                  'feedforward_dropout': self.feedforward_dropout,
                  'feedforward_multiplier': self.feedforward_multiplier,
                  'norm_epsilon': self.norm_epsilon}
        return config

    @classmethod
    def from_config(cls, config):
        # we need to override the from_config method because the parent
        # is layerflow version of Encoder which tried to extract
        # the value against `transformer_layers` key to deserialize
        # which causes KeyError
        return cls(**config)