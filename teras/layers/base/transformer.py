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

    Reference(s):
        TabTransformer: https://arxiv.org/abs/2012.06678
        SAINT: https://arxiv.org/abs/2106.01342
        FT-Transformer: https://arxiv.org/abs/2106.11959

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
