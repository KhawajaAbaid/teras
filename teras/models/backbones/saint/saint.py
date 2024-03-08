import keras
from teras.models.backbones.backbone import Backbone
from teras.layers.saint.encoder_layer import SAINTEncoderLayer
from teras.layers.saint.embedding import SAINTEmbedding
from teras.layers.cls_token import CLSToken
from teras.layers.cls_token_extraction import CLSTokenExtraction
from teras.api_export import teras_export


@teras_export("teras.model.SAINTBackbone")
class SAINTBackbone(Backbone):
    """
    SAINT Backbone based on the SAINT architecture proposed in the paper,
    "SAINT: Improved Neural Networks for Tabular Data".

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        input_dim: int, dimensionality of the input dataset. i.e. the
            number of features in the dataset.
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
        embedd_inputs: bool, whether to use `SAINTEmbedding` layer to
            create emebddings of inputs. Defaults to `True` as this is
            what we want. but when pretraining we want to use the
            `SAINTBackbone` as encoder only which expects embeddings as
            inputs, so that's when we set this parameter to `False`.
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
                 embedd_inputs: bool = True,
                 **kwargs):
        inputs = keras.layers.Input((input_dim,), name="inputs")
        x = inputs
        if embedd_inputs:
            x = SAINTEmbedding(
                embedding_dim=embedding_dim,
                cardinalities=cardinalities)(x)
        x = CLSToken(embedding_dim=embedding_dim)(x)
        for i in range(num_layers):
            x = SAINTEncoderLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                attention_dropout=attention_dropout,
                feedforward_dropout=feedforward_dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                name=f"saint_encoder_layer_{i}")(x)
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
        self.embedd_inputs = embedd_inputs

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
            "embedd_inputs": self.embedd_inputs,
        })
        return config
