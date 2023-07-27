from tensorflow import keras
from teras.utils import get_normalization_layer
from teras.utils.types import (NormalizationType,
                               ActivationType)


@keras.saving.register_keras_serializable(package="teras.layers.rtdl_resnet")
class RTDLResNetBlock(keras.layers.Layer):
    """
    The ResNet block proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        units: ``int``, default 64,
            Dimensionality of the hidden layer in the ResNet block.

        dropout_hidden: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the hidden dense layer.

        dropout_out: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the output dense layer.

        activation_hidden: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the hidden layer.

        activation_out: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the output layer.

        normalization: ``str`` or ``keras.layers.Layer``, default "batch",
            Normalization layer to normalize the inputs to the RestNet block.

        use_skip_connection: ``bool``, default True,
            Whether to use the skip connection.
    """
    def __init__(self,
                 units: int = 64,
                 dropout_hidden: float = 0.,
                 dropout_out: float = 0.,
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = "relu",
                 normalization: NormalizationType = "batch",
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
        self.dense_hidden = keras.layers.Dense(units=self.units,
                                               activation=self.activation_hidden,
                                               name="resnet_block_dense_hidden")
        self.dropout_hidden = keras.layers.Dropout(self.dropout_hidden,
                                                   name="resnet_block_dropout_hidden")
        self.dropout_out = keras.layers.Dropout(self.dropout_out,
                                                name="resnet_block_dropout_out")
        self.add = keras.layers.Add(name="resnet_block_add")
        self.dense_out = None

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.dense_out = keras.layers.Dense(units=input_dim,
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

        if isinstance(self.activation_hidden, str):
            activation_hidden_serialized = self.activation_hidden
        else:
            activation_hidden_serialized = keras.layers.serialize(self.activation_hidden)

        if isinstance(self.activation_out, str):
            activation_out_serialized = self.activation_out
        else:
            activation_out_serialized = keras.layers.serialize(self.activation_out)

        if isinstance(self.normalization, str):
            normalization_serialized = self.normalization
        else:
            normalization_serialized = keras.layers.serialize(self.normalization)

        config.update({'units': self.units,
                       'dropout_hidden': self.dropout_hidden,
                       'dropout_out': self.dropout_out,
                       'activation_hidden': activation_hidden_serialized,
                       'activation_out': activation_out_serialized,
                       'normalization': normalization_serialized,
                       'use_skip_connection': self.use_skip_connection,
                       })
        return config
