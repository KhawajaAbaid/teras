import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers.common.head import (ClassificationHead as BaseClassificationHead,
                                      RegressionHead as BaseRegressionHead)
from typing import Union, List, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
LAYER_OR_STR = Union[keras.layers.Layer, str]


class ColumnEmbedding(layers.Layer):
    """
    ColumnEmbedding layer as proposed by Xin Huang et al. in the paper
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        num_categorical_features: `int`,
            Number of categorical features in the dataset.
        embedding_dim: `int`, default 32,
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
        new_config = {'num_categorical_features': self.num_categorical_features,
                      'embedding_dim': self.embedding_dim,
                      }
        config.update(new_config)
        return config


class ClassificationHead(BaseClassificationHead):
    """
    Classification head for the TabTransformer Classifier architecture.

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        units_values: `List[int] | Tuple[int]`, default (64, 32),
            For each value in the sequence,
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the ClassificationHead.
        activation_hidden: default "relu",
            Activation function to use in hidden dense layers.
        activation_out:
            Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.
        normalization: `Layer | str`, default "batch",
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_classes: int = 2,
                 units_values: LIST_OR_TUPLE = (64, 32),
                 activation_hidden="relu",
                 activation_out=None,
                 normalization: LAYER_OR_STR = "batch",
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out,
                         normalization=normalization,
                         **kwargs)


class RegressionHead(BaseRegressionHead):
    """
    Regression head for the TabTransformer Regressor architecture.

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs to predict.
        units_values: `List[int] | Tuple[int]`, default (64, 32),
            For each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the RegressionHead.
        activation_hidden: default "relu",
            Activation function to use in hidden dense layers.
        normalization: `Layer | str`, default "batch",
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 units_values: LIST_OR_TUPLE = (64, 32),
                 activation_hidden="relu",
                 normalization: LAYER_OR_STR = "batch",
                 **kwargs):
        super().__init__(num_outputs=num_outputs,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         normalization=normalization,
                         **kwargs)
