import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers.common.transformer import (ClassificationHead as BaseClassificationHead,
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
