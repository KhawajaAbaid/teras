import keras
from teras.models.blueprints.backbone import Backbone
from teras.layers.transformer.encoder_layer import TransformerEncoderLayer
from teras.api_export import teras_export


@teras_export("teras.models.TransformerEncoderBackbone")
class TransformerEncoderBackbone(Backbone):
    """
    Transformer Encoder model as proposed in the "Attention is all you
    need" paper.

    Reference(s):
        https://arxiv.org/abs/1706.03762

    Args:
        input_dim: int, dimensionality of the input dataset. i.e. the
            number of features in the dataset.
        num_layers: int, number of `TransformerEncoderLayer`s to use in
            the encoder.
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
                 input_dim: int,
                 num_layers: int,
                 embedding_dim: int,
                 num_heads: int = 8,
                 feedforward_dim: int = None,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 layer_norm_epsilon: float = 1e-5,
                 **kwargs):
        if num_layers < 1:
            raise ValueError(
                f"`num_layers` must be 1 or greater. Received {num_layers}")
        inputs = keras.layers.Input(shape=(input_dim, embedding_dim))
        x = inputs
        for i in range(num_layers):
            x = TransformerEncoderLayer(
                                embedding_dim=embedding_dim,
                                num_heads=num_heads,
                                feedforward_dim=feedforward_dim,
                                attention_dropout=attention_dropout,
                                feedforward_dropout=feedforward_dropout,
                                layer_norm_epsilon=layer_norm_epsilon,
                                name=f"transformer_encoder_layer_{i+1}")(x)
        outputs = x
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "num_layers": self.num_layers,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "attention_dropout": self.attention_dropout,
            "feedforward_dropout": self.feedforward_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon
        })
        return config
