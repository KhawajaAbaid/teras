from tensorflow import keras
from teras.utils import get_normalization_layer
from teras.utils.types import (UnitsValuesType,
                               NormalizationType,
                               ActivationType)


@keras.saving.register_keras_serializable(package="teras.layers.common")
class RegressionHead(keras.layers.Layer):
    """
    Regression Head to use on top of the architectures for regression.

    Args:
        num_outputs: ``int``, default 1,
            Number of regression outputs to predict.

        units_values: ``List[int]`` or ``Tuple[int]``, default (64, 32),
            For each value in the sequence a hidden layer of that dimension
            preceded by a normalization layer (if specified) is
            added to the ``RegressionHead``.

        activation_hidden: ``callable`` or ``str``, default "relu",
            Activation function to use in hidden dense layers.

        normalization: ``keras.layers.Layer`` or ``str``, default "batch",
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 units_values: UnitsValuesType = (64, 32),
                 activation_hidden: ActivationType = "relu",
                 normalization: NormalizationType = "batch",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.units_values = units_values
        self.activation_hidden = activation_hidden
        self.normalization = normalization

        self.hidden_block = None
        if self.units_values is not None:
            self.hidden_block = keras.models.Sequential(name="inner_head")
            for units in self.units_values:
                if self.normalization is not None:
                    self.hidden_block.add(get_normalization_layer(self.normalization))
                self.hidden_block.add(keras.layers.Dense(units,
                                                         activation=self.activation_hidden))
        self.output_layer = keras.layers.Dense(self.num_outputs)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        outputs = self.output_layer(x)
        return outputs

    def get_config(self):
        config = super().get_config()

        if isinstance(self.activation_hidden, str):
            activation_hidden_serialized = self.activation_hidden
        else:
            activation_hidden_serialized = keras.layers.serialize(self.activation_hidden)

        if isinstance(self.normalization, str):
            normalization_serialized = self.normalization
        else:
            normalization_serialized = keras.layers.serialize(self.normalization)

        config.update({'num_outputs': self.num_outputs,
                       'units_values': self.units_values,
                       'activation_hidden': activation_hidden_serialized,
                       'normalization': normalization_serialized}
                      )
        return config


@keras.saving.register_keras_serializable(package="teras.layers.common")
class ClassificationHead(keras.layers.Layer):
    """
    Classification head to use on top of the architectures for classification.

    Args:
        num_classes: ``int``, default 2,
            Number of classes to predict.

        units_values: ``List[int]`` or ``Tuple[int]``, default (64, 32),
            For each value in the sequence a hidden layer of that dimension
            preceded by a normalization layer (if specified) is
            added to the ``ClassificationHead``.

        activation_hidden: ``callable``, ``keras.layers.Layer``, or ``str``, default "relu",
            Activation function to use in hidden dense layers.

        activation_out: ``callable``, ``keras.layers.Layer``, or ``str``,
            Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.

        normalization: ``keras.layers.Layer`` or ``str``, default "batch",
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_classes: int = 2,
                 units_values: UnitsValuesType = (64, 32),
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = None,
                 normalization: NormalizationType = "batch",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.units_values = units_values
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = "sigmoid" if self.num_classes == 1 else "softmax"
        self.normalization = normalization

        self.hidden_block = None
        if self.units_values is not None:
            self.hidden_block = keras.models.Sequential(name="inner_head")
            for units in self.units_values:
                if self.normalization is not None:
                    self.hidden_block.add(get_normalization_layer(self.normalization))
                self.hidden_block.add(keras.layers.Dense(units,
                                                         activation=self.activation_hidden))
        self.output_layer = keras.layers.Dense(self.num_classes,
                                               activation=self.activation_out)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        outputs = self.output_layer(x)
        return outputs

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

        config.update({'num_classes': self.num_classes,
                       'units_values': self.units_values,
                       'activation_hidden': activation_hidden_serialized,
                       'activation_out': activation_out_serialized,
                       'normalization': normalization_serialized}
                      )
        return config
