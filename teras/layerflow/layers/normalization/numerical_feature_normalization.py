import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.normalization")
class NumericalFeatureNormalization(keras.layers.Layer):
    """
    NumericalFeatureNormalization layer that applies specified
    type of normalization over the numerical features only.

    Args:
        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this metadata dictionary by calling
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> features_metadata = get_features_metadata_for_embedding(dataset,
                ..                                                          categorical_features,
                ..                                                          numerical_features)

        normalization: ``layers.Layer``, default ``LayerNormalization``,
            Normalization to apply over the numerical features.
    """
    def __init__(self,
                 features_metadata: dict,
                 normalization: keras.layers.Layer = keras.layers.LayerNormalization(),
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

        self._numerical_features_names = list(self.features_metadata["numerical"].keys())
        self._numerical_features_idx = list(self.features_metadata["numerical"].values())

    def call(self, inputs):
        numerical_features = tf.gather(inputs,
                                       indices=self._numerical_features_idx,
                                       axis=1)
        numerical_features = self.normalization(numerical_features)
        return numerical_features

    def get_config(self):
        config = super().get_config()
        config.update({"features_metadata": self.features_metadata,
                       "normalization": keras.layers.serialize(self.normalization)})
        return config

    @classmethod
    def from_config(cls, config):
        features_metadata = config.pop("features_metadata")
        normalization = keras.layers.deserialize(config.pop("normalization"))
        return cls(features_metadata, normalization=normalization, **config)
