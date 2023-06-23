from teras.config.base import BaseConfig


class SAINTConfig(BaseConfig):
    """Config class for SAINT architecture to hold the
    default values for the common parameters of SAINT
    architecture in one place."""
    embedding_dim: int = 32
    numerical_embedding_hidden_dim: int = 16
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    num_inter_sample_attention_heads: int = 8
    attention_dropout: float = 0.1
    inter_sample_attention_dropout: float = 0.1
    feedforward_dropout: float = 0.1
    feedforward_multiplier: int = 4
    norm_epsilon: float = 1e-6
    encode_categorical_values: bool = True
    apply_attention_to_features: bool = True
    apply_attention_to_rows: bool = True
