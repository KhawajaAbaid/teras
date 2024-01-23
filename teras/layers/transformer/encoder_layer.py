import keras
from teras.layers.transformer.feedforward import TransformerFeedForward
from teras.api_export import teras_export


@teras_export("teras.layers.TransformerEncoderLayer")
class TransformerEncoderLayer(keras.layers.Layer):
    """
    Transformer Encoder Layer as proposed in the original Transformer
    architecture in the "Attention is all you need" paper.

    This is the layer that makes up the encoder in the architecture.
    This is made up of `MultiHeadAttention` and `TransformerFeedForward`
    layers.

    Reference(s):
        https://arxiv.org/abs/1706.03762

    Args:
        embedding_dim: int, dimensionality of the embeddings used
            by the model. It is also referred to as the `d_model` or
            model dimensionality.
        num_heads: int, number of attention heads to use in the
            `MultiHeadAttention` layer.
        feedforward_dim: int, hidden dimensionality to use in the
            `TransformerFeedForward` layer.
        attention_dropout: float, dropout value to use in the
        `MultiHeadAttention` layer. Defaults to 0.
        feedforward_dropout: float, dropout value to use in the
            `TransformerFeedForward` layer. Defaults to 0.
        layer_norm_epsilon: float, epsilon value to use in the
            `LayerNormalization` layer. Defaults to 1e-5.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int = 8,
                 feedforward_dim: int = None,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 layer_norm_epsilon: float = 1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim,
            dropout=attention_dropout
        )
        self.feedforward = TransformerFeedForward(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.feedforward_dim,
            dropout=self.feedforward_dropout
        )
        self.add_1 = keras.layers.Add()
        self.add_2 = keras.layers.Add()
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )

    def call(self, inputs):
        residue = inputs
        x = self.attention(inputs, inputs)
        x = self.add_1([x, residue])
        x = self.layer_norm_1(x)
        residue = x
        x = self.feedforward(x)
        x = self.add_2([x, residue])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "attention_dropout": self.attention_dropout,
            "feedforward_dropout": self.feedforward_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon
        })
        return config
