import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class MultiHeadInterSampleAttention(keras.layers.Layer):
    """
    MultiHeadInterSampleAttention layer as proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    Unlike the usual MultiHeadAttention layer, this MultiHeadInterSampleAttention layer,
    as the name enunciates, applies attention over samples/rows instead of features/columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_heads: `int`, default 8,
            Number of Attention heads to use
        key_dim: `int`, default 32,
            Key dimensionality for attention.
        dropout: `float`, default 0.1,
            Dropout rate to use.
    """
    def __init__(self,
                 num_heads: int = 8,
                 key_dim: int = 32,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.multi_head_attention = keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                                    key_dim=self.key_dim,
                                                                    dropout=dropout,
                                                                    **kwargs)

    def call(self, inputs):
        # Expected inputs shape: (b, n, d)
        # b: batch_size, n: num_features, d: embedding_dim
        x = inputs
        x = tf.reshape(x, shape=(1,
                                 tf.shape(x)[0],
                                 tf.shape(x)[1] * tf.shape(x)[2]))
        x = self.multi_head_attention(x, x)
        x = tf.reshape(x, shape=tf.shape(inputs))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads,
                       'key_dim': self.key_dim,
                       'dropout': self.dropout,
                       })
        return config
