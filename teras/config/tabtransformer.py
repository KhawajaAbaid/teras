from teras.config.base import BaseConfig


class TabTransformerConfig(BaseConfig):
    embedding_dim: int = 32
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    attention_dropout: float = 0.
    feedforward_dropout: float = 0.
    feedforward_multiplier: int = 4
    norm_epsilon: float = 1e-6
    use_column_embedding: bool = True
    categorical_features_vocabulary: dict = None
    encode_categorical_values: bool = True
