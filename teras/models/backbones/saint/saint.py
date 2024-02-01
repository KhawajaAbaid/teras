import keras
from teras.models.backbones.backbone import Backbone
from teras.layers.saint.encoder_layer import SAINTEncoderLayer
from teras.layers.saint.embedding import SAINTEmbedding
from teras.layers.cls_token import CLSToken
from teras.layers.cls_token_extraction import CLSTokenExtraction
from teras.api_export import teras_export


@teras_export("teras.model.SAINTBackbone")
class SAINTBackbone(Backbone):
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
        inputs = keras.layers.Input((input_dim,), name="inputs")
        x = SAINTEmbedding(
            embedding_dim=embedding_dim,
            cardinalities=cardinalities)(inputs)
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
