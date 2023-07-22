import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.tabtransformer")
class TabTransformerColumnEmbedding(keras.layers.Layer):
    """
    Column Embedding layer as proposed by Xin Huang et al. in the paper
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        num_categorical_features: ``int``,
            Number of categorical features in the dataset.

        embedding_dim: ``int``, default 32,
            Dimensionality of the embedded features
    """
    def __init__(self,
                 num_categorical_features: int,
                 embedding_dim=32,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_categorical_features = num_categorical_features
        self.column_embedding = keras.layers.Embedding(input_dim=self.num_categorical_features,
                                                       output_dim=self.embedding_dim)
        self.column_indices = tf.range(start=0,
                                       limit=self.num_categorical_features,
                                       delta=1)
        self.column_indices = tf.cast(self.column_indices, dtype="float32")

    def call(self, inputs):
        """
        Args:
            inputs: Embeddings of categorical features encoded by CategoricalFeatureEmbedding layer
        """
        return inputs + self.column_embedding(self.column_indices)

    def get_config(self):
        config = super().get_config()
        config.update({'num_categorical_features': self.num_categorical_features,
                      'embedding_dim': self.embedding_dim,
                      })
        return config

    @classmethod
    def from_config(cls, config):
        num_categorical_features = config.pop("num_categorical_features")
        return cls(num_categorical_features, **config)

