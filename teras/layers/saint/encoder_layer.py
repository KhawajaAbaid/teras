import keras
from teras.layers.transformer.encoder_layer import TransformerEncoderLayer
from teras.layers.transformer.feedforward import TransformerFeedForward
from teras.layers.saint.multi_head_inter_sample_attention import SAINTMultiHeadInterSampleAttention
from teras.api_export import teras_export


@teras_export("teras.layers.SAINTEncoderLayer")
class SAINTEncoderLayer(keras.layers.Layer):
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

        # ====== Self Attention Block ========
        self.self_attention_block = TransformerEncoderLayer(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            feedforward_dim=self.feedforward_dim,
            attention_dropout=self.attention_dropout,
            feedforward_dropout=self.feedforward_dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            name="Self_Attention_Block",
        )

        # ====== Inter Sample Attention Block ========
        self.inter_sample_attention = SAINTMultiHeadInterSampleAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim // self.num_heads,   # ref: paper
            value_dim=None,
            dropout=self.attention_dropout,
        )
        self.isab_feed_forward = TransformerFeedForward(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.feedforward_dim,
            activation="gelu",
            dropout=feedforward_dropout
        )
        self.isab_norm_1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self.isab_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

    def call(self, inputs):
        # ====== Self Attention Block ========
        x = self.self_attention_block(inputs)
        # ====== Inter Sample Attention Block ========
        residue = x
        x = self.inter_sample_attention(x)
        x = keras.layers.add([x, residue])
        x = self.isab_norm_1(x)
        residue = x
        x = self.isab_feed_forward(x)
        x = keras.layers.add([x, residue])
        x = self.isab_norm_2(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "attention_dropout": self.attention_dropout,
            "feedforward_dropout": self.feedforward_dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })