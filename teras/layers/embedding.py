import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
        self.embedding_dim = embedding_dim
        self.lookup_tables, self.embedding_layers = self._get_lookup_tables_and_embedding_layers()
        self.concat = layers.Concatenate(axis=1)

    def _get_lookup_tables_and_embedding_layers(self):
        """Lookup tables and embedding layers for each categorical feature"""
        lookup_tables = {}
        embedding_layers = {}
        for feature, vocabulary in self.categorical_features_vocabulary.items():
            # Lookup Table to convert string values to integer indices
            lookup = layers.StringLookup(vocabulary=vocabulary,
                                         mask_token=None,
                                         num_oov_indices=0,
                                         output_mode="int"
                                         )
            lookup_tables[feature] = lookup

            # Embedding layers map the integer representations of categorical values
            # to dense vectors of `embedding_dim` dimensionality,
            # which in fancier lingo are called `embedddings`.
            embedding = layers.Embedding(input_dim=len(vocabulary),
                                         output_dim=self.embedding_dim)
            embedding_layers[feature] = embedding

        return lookup_tables, embedding_layers

    def call(self, inputs):
        # Encode and embedd categorical features
        categorical_features_embeddings = []
        for feature_id, feature in enumerate(self.categorical_features_vocabulary):
            lookup = self.lookup_tables[feature]
            embedding = self.embedding_layers[feature]
            # Convert string input values to integer indices
            feature = tf.expand_dims(inputs[feature], 1)
            encoded_feature = lookup(feature)
            # Convert index values to embedding representations
            encoded_feature = embedding(encoded_feature)
            categorical_features_embeddings.append(encoded_feature)

        categorical_features_embeddings = self.concat(categorical_features_embeddings)
        return categorical_features_embeddings
