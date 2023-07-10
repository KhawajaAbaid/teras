import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List


LIST_OF_STR = List[str]


class CategoricalFeatureEmbedding(layers.Layer):
    """
    CategoricalFeatureEmbedding layer that encodes categorical features into
    categorical feature embeddings.

    Args:
        categorical_features_metadata: `dict`, A dictionary of metadata of categorical features.
            It is simply a mapping where for each categorical feature, the feature name maps
            to a tuple of feature index and a list of unique values (a.k.a. Vocabulary) in the feature.
            You can get this metadata dictionary by calling
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> features_metadata = get_features_metadata_for_embedding(dataset,
                                                                            categorical_features,
                                                                            numerical_features)
            and then accessing its `categorical` key as follows:
                >>> categorical_features_metadata = features_metadata["categorical"]
        embedding_dim: `int`, default 32, Dimensionality of the embeddings
            to which categorical features should be mapped.
        encode: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
                 categorical_features_metadata: dict,
                 embedding_dim: int = 32,
                 encode: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_features_metadata = categorical_features_metadata
        self.embedding_dim = embedding_dim
        self.encode = encode

        self.lookup_tables, self.embedding_layers = self._get_lookup_tables_and_embedding_layers()
        self.concat = layers.Concatenate(axis=1)
        self._is_data_in_dict_format = False
        self._is_first_batch = True
        self._num_categorical_features = len(categorical_features_metadata)

    def _get_lookup_tables_and_embedding_layers(self):
        """Lookup tables and embedding layers for each categorical feature"""
        lookup_tables = []
        embedding_layers = []
        for feature_name, (feature_idx, vocabulary) in self.categorical_features_metadata.items():
            # If the categorical values in the dataset haven't been encoded in preprocessing step
            # then encode them first. The `encode` parameter lets user specify if we need to encode
            # the categorical values or not.
            if self.encode:
                # Lookup Table to convert string values to integer indices
                lookup = layers.StringLookup(vocabulary=vocabulary,
                                             mask_token=None,
                                             num_oov_indices=0,
                                             output_mode="int"
                                             )
                lookup_tables.append(lookup)

            # Embedding layers map the integer representations of categorical values
            # to dense vectors of `embedding_dim` dimensionality,
            # which in fancier lingo are called `embeddings`.
            embedding = layers.Embedding(input_dim=len(vocabulary),
                                         output_dim=self.embedding_dim)
            embedding_layers.append(embedding)

        return lookup_tables, embedding_layers

    def call(self, inputs):
        # Find the dataset's format - is it either in dictionary format.
        # If inputs is an instance of dict, it's in dictionary format
        # If inputs is an instance of tuple, it's in array format
        if self._is_first_batch:
            if isinstance(inputs, dict):
                self._is_data_in_dict_format = True
            self._is_first_batch = False

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
            if self._is_data_in_dict_format:
                feature = tf.expand_dims(inputs[feature_name], 1)
            else:
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
        if tf.rank(categorical_feature_embeddings) == 3:
            categorical_feature_embeddings = tf.transpose(categorical_feature_embeddings, perm=[1, 0, 2])
        else:
            categorical_feature_embeddings = tf.transpose(categorical_feature_embeddings)
        return categorical_feature_embeddings

    def get_config(self):
        config = super().get_config()
        new_config = {'categorical_features_metadata': self.categorical_features_metadata,
                      'embedding_dim': self.embedding_dim,
                      'encode': self.encode}
        config.update(new_config)
        return config
