from tensorflow import keras
from teras.utils import get_normalization_layer
from teras.layerflow.layers.normalization import NumericalFeatureNormalization


@keras.saving.register_keras_serializable(package="teras.layers")
class NumericalFeatureNormalization(NumericalFeatureNormalization):
    """
    NumericalFeatureNormalization layer that applies specified
    type of normalization over the numerical features only.


    Args:
        features_metadata: ``dict``,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this metadata dictionary by calling
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> features_metadata = get_features_metadata_for_embedding(dataset,
                ..                                                            categorical_features,
                ..                                                            numerical_features)
        normalization_type: ``str``, default "layer",
            Normalization type to apply over the numerical features.
            By default, ``LayerNormalization`` is applied.
            Allowed values ["layer", "batch", "unit", "group"] or their full names,
            e.g. "LayerNormalization".
            Note: The names are case-insensitive.
    """
    def __init__(self,
                 features_metadata: dict,
                 normalization_type: str = "layer",
                 **kwargs):
        normalization = get_normalization_layer(normalization_type)
        super().__init__(features_metadata=features_metadata,
                         normalization=normalization,
                         **kwargs)

        self.features_metadata = features_metadata
        self.normalization_type = normalization_type

    def get_config(self):
        config = {"name":self.name,
                  "trainable": self.trainable,
                  "features_metadata": self.features_metadata,
                  "normalization_type": self.normalization_type}
        return config
