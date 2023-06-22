from tensorflow import keras
from typing import Union
from teras.layers import (RTDLResNetBlock,
                          RTDLResNetClassificationHead,
                          RTDLResNetRegressionHead)


LayerType = Union[str, keras.layers.Layer]


class ResNetClassifier(keras.Model):
    """
    The ResNet Classifier model based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: Number of classes to predict
        num_blocks: Number of ResNet blocks to use.
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
        normalization_block: Normalization layer to normalize
            the inputs to the RestNet block. Defaults to BatchNormalization.
        use_skip_connection: Whether to use the skip connection in the ResNet block
        activation_out: Output activation, by default, sigmoid is used for binary
            while softmax is used for multiclass classification
        normalization_head: Normalization layer to use in the classification head
    """
    def __init__(self,
                 num_classes: int = None,
                 num_blocks: int = 1,
                 main_dim: int = None,
                 hidden_dim: int = None,
                 dropout_main: float = 0.,
                 dropout_hidden: float = 0.,
                 activation_main: LayerType = "relu",
                 activation_hidden: LayerType = "relu",
                 normalization_block: LayerType = "BatchNormalization",
                 use_skip_connection: bool = True,
                 activation_out: LayerType = None,
                 normalization_head: LayerType = "BatchNormalization",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.num_blocks = num_blocks
        self.main_dim = main_dim
        self.hidden_dim = hidden_dim
        self.dropout_main = dropout_main
        self.dropout_hidden = dropout_hidden
        self.normalization_block = normalization_block
        self.activation_main = activation_main
        self.activation_hidden = activation_hidden
        self.use_skip_connection = use_skip_connection
        self.activation_out = activation_out
        self.normalization_head = normalization_head

        self.head = RTDLResNetClassificationHead(num_classes=self.num_classes,
                                                 activation_out=self.activation_out,
                                                 normalization=self.normalization_head)

    def build(self, input_shape):
        # Keeping in accordance with the official implementation,
        # when the user doesn't specify main_dim, it will be
        # set equal to the input dimensionality
        if self.main_dim is None:
            self.main_dim = input_shape[1]

        self.first_layer = keras.layers.Dense(self.main_dim)

        self.blocks = keras.models.Sequential(
            [
                RTDLResNetBlock(
                    hidden_dim=self.hidden_dim,
                    main_dim=self.main_dim,
                    dropout_hidden=self.dropout_hidden,
                    dropout_main=self.dropout_main,
                    activation_hidden=self.activation_hidden,
                    activation_main=self.activation_main,
                    normalization=self.normalization_block,
                    use_skip_connection=self.use_skip_connection

                )
                for _ in range(self.num_blocks)
            ], name="ResNetRTDL_Body"
        )

    def call(self, inputs):
        x = self.first_layer(inputs)
        x = self.blocks(x)
        x = self.head(x)
        return x


class ResNetRegressor(keras.Model):
    """
    The ResNet Classifier model based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        units_out: Number of regression outputs
        num_blocks: Number of ResNet blocks to use.
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
        normalization_block: Normalization layer to normalize
            the inputs to the RestNet block. Defaults to BatchNormalization.
        use_skip_connection: Whether to use the skip connection in the ResNet block
        normalization_head: Normalization layer to use in the classification head
    """

    def __init__(self,
                 units_out: int = 1,
                 num_blocks: int = 1,
                 main_dim: int = None,
                 hidden_dim: int = None,
                 dropout_main: float = 0.,
                 dropout_hidden: float = 0.,
                 activation_main: LayerType = "relu",
                 activation_hidden: LayerType = "relu",
                 normalization_block: LayerType = "BatchNormalization",
                 use_skip_connection: bool = True,
                 normalization_head: LayerType = "BatchNormalization",
                 **kwargs):
        super().__init__(**kwargs)
        self.units_out = units_out
        self.num_blocks = num_blocks
        self.main_dim = main_dim
        self.hidden_dim = hidden_dim
        self.dropout_main = dropout_main
        self.dropout_hidden = dropout_hidden
        self.normalization_block = normalization_block
        self.activation_main = activation_main
        self.activation_hidden = activation_hidden
        self.use_skip_connection = use_skip_connection
        self.normalization_head = normalization_head

        self.head = RTDLResNetRegressionHead(units_out=self.units_out,
                                             normalization=self.normalization_head)

    def build(self, input_shape):
        # Keeping in accordance with the official implementation,
        # when the user doesn't specify main_dim, it will be
        # set equal to the input dimensionality
        if self.main_dim is None:
            self.main_dim = input_shape[1]

        self.first_layer = keras.layers.Dense(self.main_dim)

        self.blocks = keras.models.Sequential(
            [
                RTDLResNetBlock(
                    hidden_dim=self.hidden_dim,
                    main_dim=self.main_dim,
                    dropout_hidden=self.dropout_hidden,
                    dropout_main=self.dropout_main,
                    activation_hidden=self.activation_hidden,
                    activation_main=self.activation_main,
                    normalization=self.normalization_block,
                    use_skip_connection=self.use_skip_connection

                )
                for _ in range(self.num_blocks)
            ], name="ResNetRTDL_Body"
        )

    def call(self, inputs):
        x = self.first_layer(inputs)
        x = self.blocks(x)
        x = self.head(x)
        return x