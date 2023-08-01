import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class SAINTNumericalFeatureEmbedding(keras.layers.Layer):
    """
    SAINTNumericalFeatureEmbedding layer based on the architecture proposed
    by Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        features_metadata: ``dict``,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      numerical_features,
                ..                                                      categorical_features)

        embedding_dim: ``int``, default 32,
            Embedding dimension is the dimensionality of the output layer or
            the dimensionality of the embeddings produced.
            (These embedding dimensions are the same used for the embedding categorical features)

        hidden_dim: ``int``, default 16,
            Hidden dimension, used by the first dense layer i.e the hidden layer whose outputs
            are later projected to the ``emebedding_dim``
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 hidden_dim: int = 16,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self._num_numerical_features = len(self.features_metadata["numerical"])
        # Need to create as many embedding layers as there are numerical features
        self.embedding_layers = []
        for _ in range(self._num_numerical_features):
            self.embedding_layers.append(
                keras.models.Sequential([
                    keras.layers.Dense(units=self.hidden_dim, activation="relu"),
                    keras.layers.Dense(units=self.embedding_dim)
                ]
                )
            )

    def call(self, inputs):
        numerical_feature_embeddings = tf.TensorArray(size=self._num_numerical_features,
                                                      dtype=tf.float32)

        for i, (feature_name, feature_idx) in enumerate(self.features_metadata["numerical"].items()):
            feature = tf.expand_dims(inputs[:, feature_idx], 1)
            embedding = self.embedding_layers[i]
            feature = embedding(feature)
            numerical_feature_embeddings = numerical_feature_embeddings.write(i, feature)

        numerical_feature_embeddings = tf.squeeze(numerical_feature_embeddings.stack())
        numerical_feature_embeddings = tf.transpose(numerical_feature_embeddings, perm=[1, 0, 2])
        numerical_feature_embeddings.set_shape((None, self._num_numerical_features, self.embedding_dim))
        return numerical_feature_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({'features_metadata': self.features_metadata,
                       'embedding_dim': self.embedding_dim,
                       'hidden_dim': self.hidden_dim,
                       })
        return config
