from tensorflow import keras
from typing import Union
from teras.layers import (RTDLResNetBlock,
                          RTDLResNetClassificationHead,
                          RTDLResNetRegressionHead)


LayerType = Union[str, keras.layers.Layer]


class RTDLResNet(keras.Model):
    """
    The ResNet model based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_blocks: `int`, default 4,
            Number of ResNet blocks to use.
        block_units: `int`, default 64,
            Dimensionality of the hidden layer in the ResNet block.
        block_dropout_hidden: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the hidden dense layer.
        block_dropout_out: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the output dense layer.
        block_activation_hidden: default "relu",
            Activation function to use in the hidden layer.
        block_activation_out: default "relu",
            Activation function to use in the output layer.
        normalization: default "BatchNormalization",
            Normalization layer to normalize the inputs to the RestNet block.
        use_skip_connection: `bool`, default True,
            Whether to use the skip connection in the ResNet block.
    """
    def __init__(self,
                 num_blocks: int = 4,
                 units: int = 64,
                 dropout_hidden: float = 0.,
                 dropout_out: float = 0.,
                 activation_hidden: LayerType = "relu",
                 activation_out: LayerType = "relu",
                 normalization: LayerType = "BatchNormalization",
                 use_skip_connection: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.units = units
        self.dropout_hidden = dropout_hidden
        self.dropout_out = dropout_out
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        self.normalization = normalization
        self.use_skip_connection = use_skip_connection

        self.resnet_blocks = keras.models.Sequential(
            [
                RTDLResNetBlock(
                    units=self.units,
                    dropout_hidden=self.dropout_hidden,
                    dropout_out=self.dropout_out,
                    activation_hidden=self.activation_hidden,
                    activation_out=self.activation_out,
                    normalization=self.normalization,
                    use_skip_connection=self.use_skip_connection

                )
                for _ in range(self.num_blocks)
            ], name="resnet_blocks"
        )

    def call(self, inputs):
        outputs = self.resnet_blocks(inputs)
        if self.head is not None:
            outputs = self.head(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'num_blocks': self.num_blocks,
                      'units': self.units,
                      'dropout_hidden': self.dropout_hidden,
                      'dropout_out': self.dropout_out,
                      'activation_hidden': self.activation_hidden,
                      'activation_out': self.activation_out,
                      'normalization': self.normalization,
                      'use_skip_connection': self.use_skip_connection,
                      }
        config.update(new_config)
        return config


class RTDLResNetClassifier(RTDLResNet):
    """
    The ResNet Classifier model based on the ResNet architecture
    proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict
        activation_out:
            Activation function to use in the output layer.
            By default, "sigmoid" is used for binary while "softmax" is used for
            multiclass classification.
        num_blocks: `int`, default 4,
            Number of ResNet blocks to use.
        units: `int`, default 64,
            Dimensionality of the hidden layer in the ResNet block.
        dropout_hidden: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the hidden dense layer of each ResNet block.
        dropout_out: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the output dense layer of each ResNet block.
        activation_hidden: default "relu",
            Activation function to use in the hidden layer of each ResNet block.
        activation_out_block: default "relu",
            Activation function to use in the output layer of each ResNet block.
        normalization: default "BatchNormalization",
            Normalization layer to normalize the inputs to the RestNet block.
        use_skip_connection: `bool`, default True,
            Whether to use the skip connection in each ResNet block.
    """
    def __init__(self,
                 num_classes: int = 2,
                 activation_out=None,
                 num_blocks: int = 4,
                 units: int = 64,
                 dropout_hidden: float = 0.,
                 dropout_out: float = 0.,
                 activation_hidden: LayerType = "relu",
                 activation_out_block: LayerType = "relu",
                 normalization: LayerType = "BatchNormalization",
                 use_skip_connection: bool = True,
                 **kwargs):
        super().__init__(num_blocks=num_blocks,
                         units=units,
                         dropout_hidden=dropout_hidden,
                         dropout_out=dropout_out,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out_block,
                         normalization=normalization,
                         use_skip_connection=use_skip_connection,
                         **kwargs)

        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = "sigmoid" if self.num_classes == 1 else "softmax"

        self.head = RTDLResNetClassificationHead(num_classes=self.num_classes,
                                                 activation_out=self.activation_out,
                                                 name="rtdl_resnet_classification_head")

    def get_config(self):
        config = super().get_config()
        new_config = {'num_classes': self.num_classes,
                      'activation_out': self.activation_out
                      }
        config.update(new_config)
        return config


class RTDLResNetRegressor(keras.Model):
    """
    The ResNet Regressor model based on the ResNet architecture
    proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs
        num_blocks: `int`, default 4,
            Number of ResNet blocks to use.
        units: `int`, default 64,
            Dimensionality of the hidden layer in the ResNet block.
        dropout_hidden: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the hidden dense layer of each ResNet block.
        dropout_out: `float`, default 0.,
            Dropout rate to use for the dropout layer that is applied
            after the output dense layer of each ResNet block.
        activation_hidden: default "relu",
            Activation function to use in the hidden layer of each ResNet block.
        activation_out_block: default "relu",
            Activation function to use in the output layer of each ResNet block.
        normalization: default "BatchNormalization",
            Normalization layer to normalize the inputs to the RestNet block.
        use_skip_connection: `bool`, default True,
            Whether to use the skip connection in each ResNet block.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 num_blocks: int = 4,
                 units: int = 64,
                 dropout_hidden: float = 0.,
                 dropout_out: float = 0.,
                 activation_hidden: LayerType = "relu",
                 activation_out_block: LayerType = "relu",
                 normalization: LayerType = "BatchNormalization",
                 use_skip_connection: bool = True,
                 **kwargs):
        super().__init__(num_blocks=num_blocks,
                         units=units,
                         dropout_hidden=dropout_hidden,
                         dropout_out=dropout_out,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out_block,
                         normalization=normalization,
                         use_skip_connection=use_skip_connection,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head = RTDLResNetRegressionHead(num_outputs=self.num,
                                             normalization=self.normalization_head)

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      }
        config.update(new_config)
        return config