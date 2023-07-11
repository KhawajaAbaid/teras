from tensorflow.keras import layers
from teras.utils import get_normalization_layer
from teras.layers.common.head import (ClassificationHead as _BaseClassificationHead,
                                      RegressionHead as _BaseRegressionHead)
from typing import Union

LIST_OR_TUPLE = Union[list, tuple]
LAYER_OR_STR = Union[layers.Layer, str]


class ResNetBlock(layers.Layer):
    """
    The ResNet block proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        units: `int`, default 64,
            Dimensionality of the hidden layer in the ResNet block.
        dropout_hidden: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the hidden dense layer.
        dropout_out: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the output dense layer.
        activation_hidden: default "relu",
            Activation function to use in the hidden layer.
        activation_out: default "relu",
            Activation function to use in the output layer.
        normalization: default "BatchNormalization",
            Normalization layer to normalize the inputs to the RestNet block.
        use_skip_connection: `bool`, default True,
            Whether to use the skip connection.
    """
    def __init__(self,
                 units: int = 64,
                 dropout_hidden: float = 0.,
                 dropout_out: float = 0.,
                 activation_hidden="relu",
                 activation_out="relu",
                 normalization: LAYER_OR_STR = "BatchNormalization",
                 use_skip_connection: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_hidden = dropout_hidden
        self.dropout_out = dropout_out
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        self.normalization = normalization
        self.use_skip_connection = use_skip_connection

        if self.normalization is not None:
            self.norm = get_normalization_layer(self.normalization)
        self.dense_hidden = layers.Dense(units=self.units,
                                         activation=self.activation_hidden,
                                         name="resnet_block_dense_hidden")
        self.dropout_hidden = layers.Dropout(self.dropout_hidden,
                                             name="resnet_block_dropout_hidden")
        self.dropout_out = layers.Dropout(self.dropout_out,
                                          name="resnet_block_dropout_out")
        self.add = layers.Add(name="resnet_block_add")

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.dense_out = layers.Dense(units=input_dim,
                                      activation=self.activation_out,
                                      name="resnet_block_dense_out")

    def call(self, inputs):
        residual = inputs
        x = inputs
        if self.normalization is not None:
            x = self.norm(x)
        x = self.dense_hidden(x)
        x = self.dropout_hidden(x)
        x = self.dense_out(x)
        x = self.dropout_out(x)
        if self.use_skip_connection:
            x = self.add([x, residual])
        return x

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'dropout_hidden': self.dropout_hidden,
                      'dropout_out': self.dropout_out,
                      'activation_hidden': self.activation_hidden,
                      'activation_out': self.activation_out,
                      'normalization': self.normalization,
                      'use_skip_connection': self.use_skip_connection,
                      }
        config.update(new_config)
        return config


class ClassificationHead(_BaseClassificationHead):
    """
    Classification Head layer for the RTDLResNetClassifier.
    It is based on the ResNet architecture proposed by the Yury Gorishniy et al.
    in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        units_values: `List[int] | Tuple[int]`, default `None`,
            If specified, for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the ClassificationHead.
        activation_hidden: default `None`,
            Activation function to use in hidden dense layers.
        activation_out: default `None`,
            Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.
        normalization: `Layer | str`, default `None`,
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_classes: int = 2,
                 units_values: LIST_OR_TUPLE = None,
                 activation_hidden=None,
                 activation_out=None,
                 normalization: LAYER_OR_STR = None,
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out,
                         normalization=normalization,
                         **kwargs)


class RegressionHead(_BaseRegressionHead):
    """
    Regression Head for the RTDLResNetRegressor.
    It is based on the ResNet architecture proposed by the Yury Gorishniy et al.
    in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs to predict.
        units_values: `List[int] | Tuple[int]`, default `None`,
            If specified, for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the RegressionHead.
        activation_hidden: default `None`,
            Activation function to use in hidden dense layers.
        normalization: `Layer | str`, default `None`,
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 units_values: LIST_OR_TUPLE = None,
                 activation_hidden=None,
                 normalization: LAYER_OR_STR = None,
                 **kwargs):
        super().__init__(num_outputs=num_outputs,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         normalization=normalization,
                         **kwargs)
