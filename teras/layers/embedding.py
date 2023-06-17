import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List


LIST_OF_STR = List[str]


class CategoricalFeaturesEmbedding(layers.Layer):
    """
    CategoricalFeaturesEmbedding layer that encodes categorical features into
    categorical feature embeddings.

    Args:
        categorical_features_vocabulary: `dict`, Vocabulary of categorical feature.
            Vocabulary is simply a dictionary where feature name serves as key and its value
            is the sorted list of all unique values in that feature.
            You can get this vocabulary by calling
            # TODO rename the function to get_categorical_features_vocabulary
            `teras.utils.get_categorical_features_vocab(dataset, categorical_features)`
        embedding_dim: `int`, default 32, Dimensionality of the embeddings
            to which categorical features should be mapped.
    """
    def __init__(self,
                 categorical_features_vocabulary: dict,
                 embedding_dim: int = 32,
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_features_vocabulary = categorical_features_vocabulary
        self.categorical_features_idx = list(self.categorical_features_vocabulary.keys())
        self.embedding_dim = embedding_dim
        # self.lookup_tables = None
        # self.embedding_layers = None
        # self._prepped = False # Flag to indicate whether lookup tables and embeddings layers have been prepared
        self.lookup_tables, self.embedding_layers = self._get_lookup_tables_and_embedding_layers()
        self.concat = layers.Concatenate(axis=1)
        self.categorical_feature_indicator = None

    def _get_lookup_tables_and_embedding_layers(self):
        """Lookup tables and embedding layers for each categorical feature"""
        # lookup_tables = {}
        # embedding_layers = {}
        lookup_tables = []
        embedding_layers = []
        for idx, (feature_name, unique_values) in self.categorical_features_vocabulary.items():
        # for feature, vocabulary in self.categorical_features_vocabulary.items():
        #     # Lookup Table to convert string values to integer indices
        #     lookup = layers.StringLookup(vocabulary=unique_values,
        #                                  mask_token=None,
        #                                  num_oov_indices=0,
        #                                  output_mode="int"
        #                                  )
        #     # lookup_tables[feature_name] = lookup
        #     lookup_tables.append(lookup)

            # Embedding layers map the integer representations of categorical values
            # to dense vectors of `embedding_dim` dimensionality,
            # which in fancier lingo are called `embedddings`.
            embedding = layers.Embedding(input_dim=len(unique_values),
                                         output_dim=self.embedding_dim)
            # embedding_layers[feature_name] = embedding
            embedding_layers.append(embedding)

        return lookup_tables, embedding_layers

    def call(self, inputs):
        # Encode and embedd categorical features
        dim = tf.shape(inputs)[1]
        if self.categorical_feature_indicator is None:
            self.categorical_feature_indicator = tf.zeros(shape=(dim,))
            indices = tf.expand_dims(tf.constant(self.categorical_features_idx), axis=1)
            self.categorical_feature_indicator = tf.tensor_scatter_nd_update(self.categorical_feature_indicator, indices, tf.ones(shape=(len(indices),)))

        # feature_embeddings = []
        feature_embeddings = tf.TensorArray(size=dim, dtype=tf.float32)
        cat_index = 0
        for idx in range(dim):
            feature = tf.expand_dims(inputs[:, idx], 1)
            # We only embedd the feature if it's categorical
            # otherwise it is passed as is.
            if self.categorical_feature_indicator[idx] == 1:
                # Congrats, it's a boy... i mean categorical.
                # feature_name = self.categorical_features_vocabulary[idx][0]
                # lookup = self.lookup_tables[feature_name]
                # lookup = self.lookup_tables[cat_index]
                # embedding = self.embedding_layers[feature_name]
                embedding = self.embedding_layers[cat_index]
                # Convert string input values to integer indices
                # feature = lookup(feature)
                # Convert index values to embedding representations
                feature = embedding(feature)
                feature = tf.squeeze(feature, axis=-1)
                cat_index += 1

            # feature_embeddings.append(feature)
            feature_embeddings = feature_embeddings.write(idx, feature)

        # feature_embeddings = self.concat(feature_embeddings)
        feature_embeddings = tf.transpose(tf.squeeze(feature_embeddings.stack()))
        return feature_embeddings
