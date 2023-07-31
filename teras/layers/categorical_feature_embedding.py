from tensorflow import keras
import numpy as np
import tensorflow as tf
from teras.utils.types import Number


@keras.saving.register_keras_serializable(package="teras.layers")
class CategoricalFeatureEmbedding(keras.layers.Layer):
    """
    CategoricalFeatureEmbedding layer that encodes categorical features into
    categorical feature embeddings.

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
                ..                                                          categorical_features,
                ..                                                          numerical_features)
        embedding_dim: ``int``, default 32,
            Dimensionality of the embeddings to which categorical features should be mapped.

        encode: ``bool``, default True,
            Whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegrerLookup`` layer.
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 encode: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        if not len(features_metadata["categorical"]) > 0:
            raise ValueError("`features_metadata` does not contain any categorical features. "
                             "Either you forgot to pass categorical features names list to the "
                             "`get_features_metadata_for_embedding` or the dataset does not contain "
                             "any categorical features to begin with. \n"
                             "In either case, "
                             "`CategoricalFeatureEmbedding` cannot be called on inputs with no categorical features. ")

        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.encode = encode

        self.categorical_features_metadata = features_metadata["categorical"]
        self.lookup_tables, self.embedding_layers = self._get_lookup_tables_and_embedding_layers()
        self._num_categorical_features = len(self.categorical_features_metadata)

    def _get_lookup_tables_and_embedding_layers(self):
        """Lookup tables and embedding layers for each categorical feature"""
        lookup_tables = []
        embedding_layers = []
        for feature_name, (feature_idx, vocabulary) in self.categorical_features_metadata.items():
            # If the categorical values in the dataset haven't been encoded in preprocessing step
            # then encode them first. The `encode` parameter lets user specify if we need to encode
            # the categorical values or not.
            if self.encode:
                if isinstance(vocabulary[0], Number):
                    # Lookup Table to map integer values to integer indices
                    lookup = keras.layers.IntegerLookup(vocabulary=vocabulary,
                                                        mask_token=None,
                                                        output_mode="int",
                                                        )
                else:
                    raise TypeError("`CategoricalFeatureEmbedding` layer can only encode values of type int "
                                    f"but received type: {type(vocabulary[0])} for feature {feature_name}.")
                lookup_tables.append(lookup)

            # Embedding layers map the integer representations of categorical values
            # to dense vectors of `embedding_dim` dimensionality,
            # which in fancier lingo are called `embeddings`.
            embedding = keras.layers.Embedding(input_dim=len(vocabulary) + 1,
                                               output_dim=self.embedding_dim)
            embedding_layers.append(embedding)

        return lookup_tables, embedding_layers

    def call(self, inputs):
        # Encode and embedd categorical features
        categorical_feature_embeddings = tf.TensorArray(size=self._num_categorical_features,
                                                        dtype=tf.float32)
        # feature_idx is the overall index of feature in the dataset
        # so it can't be used to retrieve lookup table and embedding layer from list
        # which are both of length equal to number of categorical features - not input dimensions
        # hence the need for current_idx
        current_idx = 0
        # We only embedd the categorical features
        for feature_name, (feature_idx, _) in self.categorical_features_metadata.items():
            feature = tf.expand_dims(inputs[:, feature_idx], 1)
            if self.encode:
                # Convert string input values to integer indices
                lookup = self.lookup_tables[current_idx]
                feature = lookup(feature)
            # Convert index values to embedding representations
            embedding = self.embedding_layers[current_idx]
            feature = embedding(feature)
            categorical_feature_embeddings = categorical_feature_embeddings.write(current_idx, feature)
            current_idx += 1
        categorical_feature_embeddings = categorical_feature_embeddings.stack()
        categorical_feature_embeddings = tf.squeeze(categorical_feature_embeddings, axis=2)
        categorical_feature_embeddings = tf.transpose(categorical_feature_embeddings, perm=[1, 0, 2])
        categorical_feature_embeddings.set_shape((None, self._num_categorical_features, self.embedding_dim))
        return categorical_feature_embeddings

    def get_config(self):
        config = super().get_config()
        new_config = {'features_metadata': self.features_metadata,
                      'embedding_dim': self.embedding_dim,
                      'encode': self.encode}
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        features_metadata = config.pop("features_metadata")
        return cls(features_metadata, **config)
