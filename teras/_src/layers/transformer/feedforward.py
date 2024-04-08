import keras
from teras._src.api_export import teras_export
from teras._src.typing import ActivationType


@teras_export("teras.layers.TransformerFeedForward")
class TransformerFeedForward(keras.layers.Layer):
    """
    Transformer Feed Forward layer as proposed in the original
    Transformers architecture, in the paper,"Attention is all you need",
    with a slight addition of optional `Dropout` layer.

    Reference(s):
        https://arxiv.org/abs/1706.03762

    Args:
        embedding_dim: int, dimensionality of embeddings being used in
            the model
        hidden_dim: int, hidden dimensionality to use. By default,
            it is four-times of the `embedding_dim`.
        activation: str or callable, activation function to use for the
            inner linear layer. Defaults to "relu",
        dropout: float, dropout rate to use for the dropout layer
            that is applied in between the two linear layer.
            Defaults to 0., because the original transformer
            architecture doesn't employ a `Dropout` layer.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int = None,
                 activation: ActivationType = "relu",
                 dropout: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim * 4 if hidden_dim is None else hidden_dim
        self.activation = activation
        self.dropout = dropout

        self.inner = keras.layers.Dense(self.hidden_dim,
                                        activation=self.activation,
                                        name="feedforward_inner")
        self.outer = keras.layers.Dense(self.embedding_dim,
                                        name="feedforward_outer")
        self.dropout_layer = keras.layers.Dropout(
                                        self.dropout,
                                        name="feedforward_dropout")

    def build(self, input_shape):
        self.inner.build(input_shape)
        hidden_shape = input_shape[:-1] + (self.hidden_dim,)
        self.outer.build(hidden_shape)

    def call(self, inputs):
        x = self.inner(inputs)
        x = self.dropout_layer(x)
        return self.outer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation,
            "dropout": self.dropout
        })
        return config
