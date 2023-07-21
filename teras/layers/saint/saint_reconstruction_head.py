from tensorflow import keras
from teras.layerflow.layers.saint.saint_reconstruction_head import (SAINTReconstructionHeadBlock as _SAINTReconstructionHeadBlockLF,
                                                                    SAINTReconstructionHead as _SAINTReconstructionHeadLF)


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class SAINTReconstructionHeadBlock(_SAINTReconstructionHeadBlockLF):
    """
    ReconstructionBlock layer that is used in constructing ReconstructionHead.
    One ReconstructionBlock is created for each feature in the dataset.
    The inputs to this layer are first mapped to hidden dimensions
    and then projected to the cardinality of the feature.

    Args:
        feature_cardinality: ``int``,
            Cardinality of the given input feature.
            For categorical features, it is equal to the number of classes
            in the feature, and for numerical features, it is equal to 1.

        hidden_dim: ``int``, default 32,
            Dimensionality of the hidden layer.

        hidden_activation: ``str``, default "relu",
            Activation function to use in the hidden layer.
    """
    def __init__(self,
                 feature_cardinality: int,
                 hidden_dim: int = 32,
                 hidden_activation="relu",
                 **kwargs):
        hidden_block = keras.layers.Dense(hidden_dim,
                                          activation=hidden_activation,
                                          name="hidden_block")
        output_layer = keras.layers.Dense(feature_cardinality,
                                          name="output_layer")
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)
        self.feature_cardinality = feature_cardinality
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'feature_cardinality': self.feature_cardinality,
                  'hidden_dim': self.hidden_dim,
                  'hidden_activation': self.hidden_activation,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        feature_cardinality = config.pop("feature_cardinality")
        return cls(feature_cardinality, **config)


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class SAINTReconstructionHead(_SAINTReconstructionHeadLF):
    """
        ReconstructionHead layer for ``SAINTPretrainer`` model.
    SAINT applies a separate single hidden layer MLP block
    (here we name it, the reconstruction block)
    with an output layer where output dimensions are equal
    to the number of categories in the case of categorical
    features and 1 in the case of numerical features.

    Args:
        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                                                                        numerical_features,
                                                                        categorical_features)
        embedding_dim: ``int``, default 32,
            Embedding dimensions being used in the pretraining model.
            Used in the computation of the hidden dimensions for each
            reconstruction (mlp) block for each feature.
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 **kwargs):
        categorical_features_metadata = features_metadata["categorical"]
        numerical_features_metadata = features_metadata["numerical"]
        num_categorical_features = len(categorical_features_metadata)
        num_numerical_features = len(numerical_features_metadata)

        # feature_cardinalities: Dimensions of each feature in the input
        # For a categorical feature, it is equal to the number of unique categories in the feature
        # For a numerical features, it is equal to 1
        features_cardinalities = []
        # recall that categorical_features_metadata dict maps feature names to a tuple of
        # feature id and unique values in the feature
        if num_categorical_features > 0:
            features_cardinalities = list(map(lambda x: len(x[1]), categorical_features_metadata.values()))
        features_cardinalities.extend([1] * num_numerical_features)

        # For the computation of denoising loss, we use a separate MLP block for each feature
        # we call the combined blocks, reconstruction heads
        reconstruction_blocks = [
            SAINTReconstructionHeadBlock(feature_cardinality=cardinality,
                                         hidden_dim=embedding_dim * 5,
                                         )
            for cardinality in features_cardinalities]

        super().__init__(reconstruction_blocks, **kwargs)

        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'features_metadata': self.features_metadata,
                  'embedding_dim': self.embedding_dim,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        # features_metadata is the only positional argument
        features_metadata = config.pop("features_metadata")
        return cls(features_metadata, **config)
