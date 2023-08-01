import tensorflow as tf
from tensorflow import keras
from teras.utils import get_normalization_layer
from teras.utils.types import NormalizationType


@keras.saving.register_keras_serializable(package="teras.layers")
class NumericalFeatureNormalization(keras.layers.Layer):
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
        normalization: ``str``, default "layer",
            Normalization type to apply over the numerical features.
            By default, ``LayerNormalization`` is applied.
            Allowed values ["layer", "batch", "unit", "group"] or their full names,
            e.g. "LayerNormalization".
            Note: The names are case-insensitive.
    """
    def __init__(self,
                 features_metadata: dict,
                 normalization: NormalizationType = "layer",
                 **kwargs):
        super().__init__(**kwargs)
        if not len(features_metadata["numerical"]) > 0:
            raise ValueError("`features_metadata` contains no numerical features. "
                             "Either you forgot to pass numerical features names list to the "
                             "`get_features_metadata_for_embedding` or the dataset does not contain "
                             "any numerical features to begin with. \n"
                             "In either case, "
                             "`NumericalFeatureNormalization` cannot be called on inputs with no numerical features. ")
        self.features_metadata = features_metadata
        self.normalization = normalization
        self.norm = get_normalization_layer(normalization)

        self._numerical_features_idx = list(self.features_metadata["numerical"].values())

    def call(self, inputs):
        numerical_features = tf.gather(inputs,
                                       indices=self._numerical_features_idx,
                                       axis=1)
        numerical_features = self.norm(numerical_features)
        return numerical_features

    def get_config(self):
        config = super().get_config()
        if isinstance(self.normalization, str):
            normalization_serialized = self.normalization
        else:
            normalization_serialized = keras.layers.serialize(self.normalization)

        config.update({'features_metadata': self.features_metadata,
                       'normalization': normalization_serialized,
                       })
        return config
