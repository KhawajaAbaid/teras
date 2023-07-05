from tensorflow.keras import layers
from typing import Union
from teras.utils import get_normalization_layer


LayerType = Union[str, layers.Layer]


# LAYER
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
                 units: int = None,
                 dropout_hidden: float = 0.,
                 dropout_out: float = 0.,
                 activation_hidden: LayerType = "relu",
                 activation_out: LayerType = "relu",
                 normalization: LayerType = "BatchNormalization",
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
        self.dropout_hidden = layers.Dropout(self.hidden_dropout_rate,
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


class ClassificationHead(layers.Layer):
    """
    The ResNet classification head based on the architecture
    implemented and proposed by the Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict
        activation_out:
            Output activation to use.
            By default, "sigmoid" is used for binary
            while "softmax" is used for multiclass classification.
        normalization: default "BatchNormalization",
            Normalization layer to apply over the inputs.
    """
    def __init__(self,
                 num_classes: int = None,
                 activation_out: LayerType = None,
                 normalization: LayerType = "BatchNormalization",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.activation_out = activation_out
        self.normalization = normalization

        if self.normalization is not None:
            self.norm = get_normalization_layer(self.normalization)

        if self.activation_out is None:
            self.activation_out = 'sigmoid' if self.num_classes == 1 else 'softmax'

        self.output_layer = layers.Dense(num_classes,
                                         activation=activation_out)

    def call(self, inputs):
        x = inputs
        if self.normalization:
            x = self.norm(x)
        x = self.output_layer(x)
        return x


class RegressionHead(layers.Layer):
    """
    The ResNet regression head based on the architecture
    implemented and proposed by the Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs
        normalization: default "BatchNormalization",
            Normalization layer to apply over the inputs.
    """

    def __init__(self,
                 num_outputs: int = 1,
                 normalization: LayerType = "BatchNormalization",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.normalization = normalization

        if self.normalization is not None:
            self.norm = get_normalization_layer(self.normalization)

        self.output_layer = layers.Dense(self.num_outputs)

    def call(self, inputs):
        x = inputs
        if self.normalization:
            x = self.norm(x)
        x = self.output_layer(x)
        return x
