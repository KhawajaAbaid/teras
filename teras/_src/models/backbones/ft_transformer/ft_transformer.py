import keras

from teras._src.layers.cls_token import CLSToken
from teras._src.layers.cls_token_extraction import CLSTokenExtraction
from teras._src.layers.ft_transformer.feature_tokenizer import \
    FTTransformerFeatureTokenizer
from teras._src.models.backbones.backbone import Backbone
from teras._src.models.backbones.transformer.encoder import \
    TransformerEncoderBackbone
from teras._src.api_export import teras_export


@teras_export("teras.models.FTTransformerBackbone")
class FTTransformerBackbone(Backbone):
    """
    FT-Transformer Encoder backbone based on the FT-Transformer
    architecture proposed in the
    "Revisiting Deep Learning Models for Tabular Data" paper.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        input_dim: int, dimensionality of the input data.
        cardinalities: list, a list cardinalities of all the features
            in the dataset in the same order as the features' occurrence.
            For numerical features, use any value <=0 as indicator at
            the corresponding index.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
        embedding_dim: int, dimensionality of the embeddings used
            by the model. It is also referred to as the `d_model` or
            model dimensionality.
        num_layers: int, number of `TransformerEncoderLayer`s to use in
            the encoder.
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
                 cardinalities: list,
                 embedding_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 feedforward_dim: int = None,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 layer_norm_epsilon: float = 1e-5,
                 **kwargs):
        inputs = keras.layers.Input(shape=(input_dim,), name="inputs")
        x = FTTransformerFeatureTokenizer(
            cardinalities=cardinalities,
            embedding_dim=embedding_dim)(inputs)
        x = CLSToken(embedding_dim=embedding_dim)(x)
        x = TransformerEncoderBackbone(
            input_dim=input_dim + 1,    # Add plus one for the cls token
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            attention_dropout=attention_dropout,
            feedforward_dropout=feedforward_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            unnormalized_layers=[0],
            pre_normalization=True)(x)
        outputs = CLSTokenExtraction()(x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.input_dim = input_dim
        self.cardinalities = cardinalities
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
            "cardinalities": self.cardinalities,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "attention_dropout": self.attention_dropout,
            "feedforward_dropout": self.feedforward_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config
