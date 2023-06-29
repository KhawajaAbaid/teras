import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ColumnEmbedding(layers.Layer):
    """
    ColumnEmbedding layer as proposed by Xin Huang et al. in the paper
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        TODO
    """
    def __init__(self,
                 embedding_dim=32,
                 num_categorical_features=None,
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

