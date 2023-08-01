import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers")
class NumericalFeaturesExtractor(keras.layers.Layer):
    """
    NumericalFeaturesExtractor layer extracts numerical features as is.
    It helps us build functional model, in case we don't want to apply
    any special layer to numerical features but still want to extract
    numerical features to concatenate them with the categorical features.

    Args:
        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            ``{feature_name: (feature_idx, vocabulary)}`` for feature in categorical features.
            ``{feature_name: feature_idx}`` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

    Raises:
        If no numerical features exist, this layer raises a ``ValueError``.
    """
    def __init__(self,
                 features_metadata: dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.features_metadata = features_metadata
        if not len(self.features_metadata["numerical"]) > 0:
            raise ValueError("`features_metadata` contains no numerical features. "
                             "Either you forgot to pass numerical features names list to the "
                             "`get_features_metadata_for_embedding` or the dataset does not contain "
                             "any numerical features to begin with. \n"
                             "In either case, "
                             "`NumericalFeaturesExtractor` cannot be called on inputs with no numerical features. ")
        self._numerical_features_idx = list(self.features_metadata["numerical"].values())

    def call(self, inputs):
        numerical_features = tf.gather(inputs,
                                       indices=self._numerical_features_idx,
                                       axis=1)
        return numerical_features
