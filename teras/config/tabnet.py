from teras.config.base import BaseConfig


class TabNetConfig(BaseConfig):
    feature_transformer_dim: int = 32
    decision_step_output_dim: int = 32
    num_decision_steps: int = 5
    num_shared_layers: int = 2
    num_decision_dependent_layers: int = 2
    relaxation_factor: float = 1.5
    batch_momentum: float = 0.9
    virtual_batch_size: int = 64
    residual_normalization_factor: float = 0.5
    epsilon = 1e-5
    categorical_features_vocabulary: dict = None
    encode_categorical_features: bool = True

