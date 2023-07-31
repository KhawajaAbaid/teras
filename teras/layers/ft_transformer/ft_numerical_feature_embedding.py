import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.ft_transformer")
class FTNumericalFeatureEmbedding(keras.layers.Layer):
    """
    Numerical Feature Emebdding layer as proposed by
    Yury Gorishniy et al. in the paper,
    Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    It simply just projects inputs with  dimensions to Embedding dimensions.
    And the only difference between this NumericalFeatureEmbedding
    and the SAINT's NumericalFeatureEmbedding layer that,
    this layer doesn't use a hidden layer. That's it.

    Reference(s):
        https://arxiv.org/abs/2106.11959

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
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Dimensionality of embeddings, must be the same as the one
            used for embedding categorical features.
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        if not len(features_metadata["numerical"]) > 0:
            raise ValueError("`features_metadata` contains no numerical features. "
                             "Either you forgot to pass numerical features names list to the "
                             "`get_features_metadata_for_embedding` or the dataset does not contain "
                             "any numerical features to begin with. \n"
                             "In either case, "
                             "`FTNumericalFeatureEmbedding` cannot be called on inputs with no numerical features. ")
        self.features_metadata = features_metadata
        self.numerical_features_metadata = self.features_metadata["numerical"]
        self.embedding_dim = embedding_dim

        self._num_numerical_features = len(self.numerical_features_metadata)
        self._numerical_features_idx = list(self.numerical_features_metadata.values())
        # Need to create as many embedding layers as there are numerical features
        self.embedding_layers = []
        for _ in range(self._num_numerical_features):
            self.embedding_layers.append(
                keras.layers.Dense(units=self.embedding_dim)
            )

    def call(self, inputs):
        numerical_feature_embeddings = tf.TensorArray(size=self._num_numerical_features,
                                                      dtype=tf.float32)

        for i, feature_idx in enumerate(self._numerical_features_idx):
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
                      'embedding_dim': self.embedding_dim})
        return config

    @classmethod
    def from_config(cls, config):
        features_metadata = config.pop("features_metadata")
        return cls(features_metadata, **config)
