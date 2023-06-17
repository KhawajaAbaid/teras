from teras.config.base import BaseConfig


class TabNetConfig(BaseConfig):
    categorical_features_vocabulary: dict = None
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


class TabNetRegressorConfig(TabNetConfig):
    num_outputs: int = 1


class TabNetClassifierConfig(TabNetConfig):
    num_classes: int = 2


class TabNetPreTrainerConfig(BaseConfig):
    data_dim: int = 1
    missing_feature_probability: float = 0.3
    categorical_features_vocabulary: dict = None
    relaxation_factor: float = 1.5
    batch_momentum: float = 0.9
    virtual_batch_size: int = 64
    residual_normalization_factor: float = 0.5
    epsilon = 1e-5
    # Encoder specific paramters
    encoder_feature_transformer_dim: int = 32
    encoder_decision_step_output_dim: int = 32
    encoder_num_decision_steps: int = 5
    encoder_num_shared_layers: int = 2
    encoder_num_decision_dependent_layers: int = 2
    # Decoder specific parameters
    decoder_feature_transformer_dim: int = 32
    decoder_decision_step_output_dim: int = 32
    decoder_num_decision_steps: int = 5
    decoder_num_shared_layers: int = 2
    decoder_num_decision_dependent_layers: int = 2

