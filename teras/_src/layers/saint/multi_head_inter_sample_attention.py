import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.SAINTMultiHeadInterSampleAttention")
class SAINTMultiHeadInterSampleAttention(keras.layers.Layer):
    """
    Multi Head Inter Sample Attention layer based on the SAINT architecture
    proposed in the "SAINT: Improved Neural Networks for Tabular Data"
    paper.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_heads: int, number of attention heads to use.
        key_dim: int, the paper proposes to use embedding_dim/num_heads
            dimensions for your key dimensionality
        value_dim: int, same value as key_dim is used by the paper.
        dropout: float, dropout value to use. Defaults to 0.

    Shapes:
        Input Shape: (batch_size, num_features, embedding_dim)
        Output Shape: (batch_size, num_features, embedding_dim)
    """
    def __init__(self,
                 num_heads: int,
                 key_dim: int,
                 value_dim: int = None,
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout

        self.multi_head_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            dropout=self.dropout,
        )

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Inputs must have the shape of rank 3"
                "(`batch_size`, `num_features`, embedding_dim`) but "
                f"received shape {input_shape} with rank "
                f"{len(input_shape)}."
            )
        input_shape = (1, input_shape[0], input_shape[1] * input_shape[2])
        self.multi_head_attention.build(input_shape, input_shape)

    def call(self, inputs):
        batch_size, num_features, embedding_dim = ops.shape(inputs)
        x = ops.reshape(inputs,
                        (1, batch_size, num_features * embedding_dim),
                        )
        x = self.multi_head_attention(x, x)
        x = ops.reshape(x,
                        (batch_size, num_features, embedding_dim))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape  # easy!

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "dropout": self.dropout
        })
