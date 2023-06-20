import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers.activations import GEGLU


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
        num_heads: `int`, default 8, Number of heads to use in the
            MultiHeadAttention layer.
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
                 num_heads: int = 8,
                 embedding_dim: int = 32,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 norm_epsilon: float = 1e-6,
                 **kwagrs):
        super().__init__(**kwagrs)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon

        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim,
            dropout=self.attention_dropout
        )
        self.skip_1 = layers.Add()
        self.layer_norm_1 = layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.feed_forward = FeedForward(self.embedding_dim)
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
