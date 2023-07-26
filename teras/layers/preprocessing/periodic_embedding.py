import tensorflow as tf
from tensorflow import keras
from typing import Literal
import math

PERIOD_INITIALIZATIONS = Literal["log-linear", "normal"]


@keras.saving.register_keras_serializable(package="teras.layers.preprocessing")
class PeriodicEmbedding(keras.layers.Layer):
    """
    PeriodicEmbedding layer for numerical features
    as proposed Yury Gorishniy et al. in the paper
    On Embeddings for Numerical Features in Tabular Deep Learning, 2022.

    Reference(s):
        https://arxiv.org/abs/2203.05556

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

        embedding_dim: ``int``, default 32,
            Dimensionality of numerical embeddings

        initialization: default "normal",
            Initialization strategy.

        sigma: ``float``, default 0.01,
            Used for coefficients initialization
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 initialization: PERIOD_INITIALIZATIONS = "normal",
                 sigma: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        if not len(features_metadata["numerical"]) > 0:
            raise ValueError("`features_metadata` contains no numerical features. "
                             "Either you forgot to pass numerical features names list to the "
                             "`get_features_metadata_for_embedding` or the dataset does not contain "
                             "any numerical features to begin with. \n"
                             "In either case, "
                             "`PeriodicEmbedding` cannot be called on inputs with no numerical features. ")

        if initialization.lower() not in ["normal", "log-linear"]:
            raise ValueError("Invalid value for initialization. Must be one of ['log-linear', 'normal']\n"
                             f"Received: {initialization}")
        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.initialization = initialization.lower()
        self.sigma = sigma

        self._numerical_features_indices = list(self.features_metadata["numerical"].values())
        self._num_numerical_features = len(self.features_metadata["numerical"])
        # The official implementation uses another variable n, that is half of the embedding dim
        self.n = self.embedding_dim // 2
        self.coefficients = None

    def build(self, input_shape):
        if self.initialization == "log-linear":
            self.coefficients = self.sigma ** (tf.range(self.n) / self.n)
            self.coefficients = tf.repeat(self.coefficients[None],
                                          repeats=self._num_numerical_features,
                                          axis=1)
        else:
            # initialization must be normal
            self.coefficients = tf.random.normal(shape=(self._num_numerical_features, self.n),
                                                 mean=0.,
                                                 stddev=self.sigma)

        self.coefficients = tf.Variable(self.coefficients)

    @staticmethod
    def cos_sin(x):
        return tf.concat([tf.cos(x), tf.sin(x)], -1)

    def call(self, inputs):
        numerical_features = tf.gather(inputs,
                                       indices=self._numerical_features_indices,
                                       axis=1)
        numerical_features = tf.cast(numerical_features, dtype=tf.float32)

        pi = tf.constant(math.pi)
        return self.cos_sin(2. * pi
                            * tf.expand_dims(self.coefficients, axis=0)
                            * tf.expand_dims(numerical_features, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({'features_metadata': self.features_metadata,
                       'embedding_dim': self.embedding_dim,
                       'initialization': self.initialization,
                       'sigma': self.sigma,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        features_metadata = config.pop("features_metadata")
        return cls(features_metadata=features_metadata,
                   **config)
