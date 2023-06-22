import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Union
from teras.utils import get_normalization_layer


LayerType = Union[str, keras.layers.Layer]


# LAYER
class ResNetBlock(keras.layers.Layer):
    """
    The ResNet block proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        main_dim: Main dimensionality. This is the dimensionality of the inputs
            expected by the ResNet block and the output of this block will be of this dimensionality.
        hidden_dim: Dimensionality of the hidden layer in the ResNet block.
        dropout_main: Dropout rate to use for the dropout layer that is applied
            after the hidden dense layer.
        dropout_main: Dropout rate to use for the dropout layer that is applied
            after the second and last 'main' dense layer.
        bias_hidden: Whether to use bias in the hidden dense layer
        bias_main: Whether to use bias in the output or main dense layer.
        activation_hidden: Activation to use bias in the hidden dense layer
        activation_main: Activation to use bias in the main dense layer
        normalization: Normalization layer to normalize
            he inputs to the RestNet block. Defaults to BatchNormalization.
        use_skip_connection: Whether to use the skip connection
    """
    def __init__(self,
                 main_dim: int = None,
                 hidden_dim: int = None,
                 dropout_hidden: float = 0.,
                 dropout_main: float = 0.,
                 bias_hidden: bool = True,
                 bias_main: bool = True,
                 activation_hidden: LayerType = "relu",
                 activation_main: LayerType = "relu",
                 normalization: LayerType = "BatchNormalization",
                 use_skip_connection: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.main_dim = main_dim
        self.hidden_dim = hidden_dim
        self.hidden_dropout_rate = dropout_hidden
        self.main_dropout_rate = dropout_main
        self.bias_hidden = bias_hidden
        self.bias_main = bias_main
        self.activation_hidden = activation_hidden
        self.activation_main = activation_main
        self.normalization = normalization
        self.use_skip_connection = use_skip_connection

        if self.normalization is not None:
            self.norm = get_normalization_layer(self.normalization)
        self.dense_hidden = layers.Dense(units=self.hidden_dim,
                                               activation=self.activation_hidden,
                                               name="resnet_block_dense_hidden")
        self.dropout_hidden = layers.Dropout(self.hidden_dropout_rate,
                                             name="resnet_block_dropout_hidden")
        self.dense_main = layers.Dense(units=self.main_dim,
                                       activation=self.activation_main,
                                       name="resnet_block_dense_main")
        self.dropout_main = layers.Dropout(self.main_dropout_rate,
                                           name="resnet_block_dropout_main")
        self.add = layers.Add(name="resnet_block_add")

    def call(self, inputs):
        residual = inputs
        x = inputs
        if self.normalization is not None:
            x = self.norm(x)
        x = self.dense_hidden(x)
        x = self.dropout_hidden(x)
        x = self.dense_main(x)
        x = self.dropout_main(x)
        if self.use_skip_connection:
            x = self.add([x, residual])
        return x


class ClassificationHead(keras.layers.Layer):
    """
    The ResNet classification head based on the architecture
    implemented and proposed by the Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: Number of classes to predict
        activation_out: Output activation to use.
            By default, sigmoid will be use for binary
            while softmax will be use for multiclass classification.
        use_bias: Whether to use bias in the output dense layer
        normalization: Normalization layer to apply over the inputs.
            Defaults to BatchNormalization.
    """
    def __init__(self,
                 num_classes: int = None,
                 activation_out: LayerType = None,
                 use_bias: bool = True,
                 normalization: LayerType = "BatchNormalization",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.activation_out = activation_out
        self.use_bias = use_bias
        self.normalization = normalization

        if self.normalization is not None:
            self.norm = get_normalization_layer(self.normalization)

        if self.activation_out is None:
            self.activation_out = 'sigmoid' if self.num_classes == 1 else 'softmax'

        self.dense_out = keras.layers.Dense(num_classes,
                                            use_bias=self.use_bias,
                                            activation=activation_out)

    def call(self, inputs):
        x = inputs
        if self.normalization:
            x = self.norm(x)
        x = self.dense_out(x)
        return x


class RegressionHead(keras.layers.Layer):
    """
    The ResNet regression head based on the architecture
    implemented and proposed by the Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        units_out: Number of regression outputs
        use_bias: Whether to use bias in the output dense layer
        normalization: Normalization layer to apply over the inputs.
            Defaults to BatchNormalization.
    """

    def __init__(self,
                 units_out: int = 1,
                 use_bias: bool = True,
                 normalization: LayerType = "BatchNormalization",
                 **kwargs):
        super().__init__(**kwargs)
        self.units_out = units_out
        self.use_bias = use_bias
        self.normalization = normalization

        if self.normalization is not None:
            self.norm = get_normalization_layer(self.normalization)

        self.dense_out = keras.layers.Dense(units_out,
                                            use_bias=self.use_bias)

    def call(self, inputs):
        x = inputs
        if self.normalization:
            x = self.norm(x)
        x = self.dense_out(x)
        return x