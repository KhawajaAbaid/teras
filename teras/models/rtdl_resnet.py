from tensorflow import keras
from teras.layers.rtdl_resnet.rtdl_resnet_block import RTDLResNetBlock
from teras.layerflow.models.rtdl_resnet import RTDLResNet as _RTDLResNetLF
from teras.utils.types import (ActivationType,
                               NormalizationType,
                               UnitsValuesType)
from teras.layers.common.head import ClassificationHead, RegressionHead


@keras.saving.register_keras_serializable(package="teras.models")
class RTDLResNet(_RTDLResNetLF):
    """
    The ResNet model based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        input_dim: ``int``,
            The dimensionality of the input dataset,
            or the number of features in the input dataset.

        num_blocks: ``int``, default 4,
            Number of ResNet blocks to use.

        block_hidden_dim: ``int``, default 64,
            Dimensionality of the hidden layer in the ResNet block.

        block_dropout_hidden: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the hidden dense layer in the ``RTDLResNetBlock`` layer.

        block_dropout_out: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the output dense layer in the ``RTDLResNetBlock`` layer.

        block_activation_hidden: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the hidden layer in the ``RTDLResNetBlock`` layer.

        block_activation_out: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the output layer in the ``RTDLResNetBlock`` layer.

        block_normalization: ``str`` or ``keras.layers.Layer``, default "batch",
            Normalization layer to normalize the inputs to the ``RTDLResNetBlock`` layer.

        use_skip_connection: ``bool``, default True,
            Whether to use the skip connection in the ``RTDLResNetBlock`` layer.
    """
    def __init__(self,
                 input_dim: int,
                 num_blocks: int = 4,
                 block_hidden_dim: int = 64,
                 block_dropout_hidden: float = 0.,
                 block_dropout_out: float = 0.,
                 block_activation_hidden: ActivationType = "relu",
                 block_activation_out: ActivationType = "relu",
                 block_normalization: NormalizationType = "BatchNormalization",
                 use_skip_connection: bool = True,
                 **kwargs):
        resnet_blocks = keras.models.Sequential(
            [
                RTDLResNetBlock(
                    units=block_hidden_dim,
                    dropout_hidden=block_dropout_hidden,
                    dropout_out=block_dropout_out,
                    activation_hidden=block_activation_hidden,
                    activation_out=block_activation_out,
                    normalization=block_normalization,
                    use_skip_connection=use_skip_connection
                )
                for _ in range(num_blocks)
            ],
            name="resnet_blocks"
        )
        super().__init__(input_dim=input_dim,
                         resnet_blocks=resnet_blocks,
                         **kwargs)
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.block_hidden_dim = block_hidden_dim
        self.block_dropout_hidden = block_dropout_hidden
        self.block_dropout_out = block_dropout_out
        self.block_activation_hidden = block_activation_hidden
        self.block_activation_out = block_activation_out
        self.block_normalization = block_normalization
        self.use_skip_connection = use_skip_connection

    def get_config(self):
        activation_hidden_serialized = self.block_activation_hidden
        if not isinstance(self.block_activation_hidden, str):
            activation_hidden_serialized = keras.layers.serialize(self.block_activation_hidden)

        activation_out_serialized = self.block_activation_out
        if not isinstance(self.block_activation_out, str):
            activation_out_serialized = keras.layers.serialize(self.block_activation_out)

        normalization_serialized = self.block_normalization
        if not isinstance(self.block_normalization, str):
            normalization_serialized = keras.layers.serialize(self.block_normalization)

        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'num_blocks': self.num_blocks,
                  'block_hidden_dim': self.block_hidden_dim,
                  'block_dropout_hidden': self.block_dropout_hidden,
                  'block_dropout_out': self.block_dropout_out,
                  'block_activation_hidden': activation_hidden_serialized,
                  'block_activation_out': activation_out_serialized,
                  'block_normalization': normalization_serialized,
                  'use_skip_connection': self.use_skip_connection,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim,
                   **config)


@keras.saving.register_keras_serializable(package="teras.models")
class RTDLResNetClassifier(RTDLResNet):
    """
    The ResNet Classifier model based on the ResNet architecture
    proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: ``int``, default 2,
            Number of classes to predict

        head_units_values: ``List[int]`` or ``Tuple[int]``, default [64, 32],
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.

        input_dim: ``int``,
            The dimensionality of the input dataset,
            or the number of features in the input dataset.

        num_blocks: ``int``, default 4,
            Number of ResNet blocks to use.

        block_hidden_dim: ``int``, default 64,
            Dimensionality of the hidden layer in the ResNet block.

        block_dropout_hidden: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the hidden dense layer in the ``RTDLResNetBlock`` layer.

        block_dropout_out: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the output dense layer in the ``RTDLResNetBlock`` layer.

        block_activation_hidden: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the hidden layer in the ``RTDLResNetBlock`` layer.

        block_activation_out: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the output layer in the ``RTDLResNetBlock`` layer.

        block_normalization: ``str`` or ``keras.layers.Layer``, default "batch",
            Normalization layer to normalize the inputs to the ``RTDLResNetBlock`` layer.

        use_skip_connection: ``bool``, default True,
            Whether to use the skip connection in the ``RTDLResNetBlock`` layer.
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 num_blocks: int = 4,
                 block_hidden_dim: int = 64,
                 block_dropout_hidden: float = 0.,
                 block_dropout_out: float = 0.,
                 block_activation_hidden: ActivationType = "relu",
                 block_activation_out: ActivationType = "relu",
                 block_normalization: NormalizationType = "batch",
                 use_skip_connection: bool = True,
                 **kwargs):
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_values,
                                  name="rtdl_resnet_classification_head")
        super().__init__(input_dim=input_dim,
                         num_blocks=num_blocks,
                         block_hidden_dim=block_hidden_dim,
                         block_dropout_hidden=block_dropout_hidden,
                         block_dropout_out=block_dropout_out,
                         block_activation_hidden=block_activation_hidden,
                         block_activation_out=block_activation_out,
                         block_normalization=block_normalization,
                         use_skip_connection=use_skip_connection,
                         head=head,
                         **kwargs)
        self.num_classes = num_classes
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'head_units_values': self.head_units_values
                       }
                      )
        return config


@keras.saving.register_keras_serializable(package="teras.models")
class RTDLResNetRegressor(RTDLResNet):
    """
    The ResNet Regressor model based on the ResNet architecture
    proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_outputs: ``int``, default 1,
            Number of regression outputs

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

        input_dim: ``int``,
            The dimensionality of the input dataset,
            or the number of features in the input dataset.

        num_blocks: ``int``, default 4,
            Number of ResNet blocks to use.

        block_hidden_dim: ``int``, default 64,
            Dimensionality of the hidden layer in the ResNet block.

        block_dropout_hidden: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the hidden dense layer in the ``RTDLResNetBlock`` layer.

        block_dropout_out: ``float``, default 0.,
            Dropout rate to use for the ``Dropout`` layer that is applied
            after the output dense layer in the ``RTDLResNetBlock`` layer.

        block_activation_hidden: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the hidden layer in the ``RTDLResNetBlock`` layer.

        block_activation_out: ``callable`` or ``str`` or ``keras.layers.Layer``, default "relu",
            Activation function to use in the output layer in the ``RTDLResNetBlock`` layer.

        block_normalization: ``str`` or ``keras.layers.Layer``, default "batch",
            Normalization layer to normalize the inputs to the ``RTDLResNetBlock`` layer.

        use_skip_connection: ``bool``, default True,
            Whether to use the skip connection in the ``RTDLResNetBlock`` layer.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 num_blocks: int = 4,
                 block_hidden_dim: int = 64,
                 block_dropout_hidden: float = 0.,
                 block_dropout_out: float = 0.,
                 block_activation_hidden: ActivationType = "relu",
                 block_activation_out: ActivationType = "relu",
                 block_normalization: NormalizationType = "batch",
                 use_skip_connection: bool = True,
                 **kwargs):
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_values,
                              name="rtdl_resnet_regression_head")
        super().__init__(input_dim=input_dim,
                         num_blocks=num_blocks,
                         block_hidden_dim=block_hidden_dim,
                         block_dropout_hidden=block_dropout_hidden,
                         block_dropout_out=block_dropout_out,
                         block_activation_hidden=block_activation_hidden,
                         block_activation_out=block_activation_out,
                         block_normalization=block_normalization,
                         use_skip_connection=use_skip_connection,
                         head=head,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_outputs': self.num_outputs,
                       'head_units_values': self.head_units_values,
                       }
                      )
        return config
