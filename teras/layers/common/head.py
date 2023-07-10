from tensorflow import keras
from tensorflow.keras import layers, models
from teras.utils import get_normalization_layer
from typing import List, Tuple, Union


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
LAYER_OR_STR = Union[keras.layers.Layer, str]


class RegressionHead(layers.Layer):
    """
    Regression head to use on top of the architectures for regression.

    Args:
        num_outputs: `int`, default 1, Number of regression outputs to predict.
        units_values: `List[int] | Tuple[int]`, default (64, 32), for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the RegressionHead.
        activation_hidden: default "relu", Activation function to use in hidden dense layers.
        normalization: `Layer | str`, default "batch", Normalization layer to use.
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
                self.hidden_block.add(layers.Dense(units,
                                                   activation=self.activation_hidden))
        self.output_layer = layers.Dense(self.num_outputs)

    # def build(self, input_shape):
    #     self.hidden_block.build(input_shape)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        outputs = self.output_layer(x)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      'units_values': self.units_values,
                      'activation_hidden': self.activation_hidden,
                      'normalization': self.normalization}
        config.update(new_config)
        return config


class ClassificationHead(layers.Layer):
    """
    Classification head to use on top of the architectures for classification.

    Args:
        num_classes: `int`, default 2, Number of classes to predict.
        units_values: `List[int] | Tuple[int]`, default (64, 32), for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the ClassificationHead.
        activation_hidden: default "relu", Activation function to use in hidden dense layers.
        activation_out: Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.
        normalization: `Layer | str`, default "batch", Normalization layer to use.
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
                self.hidden_block.add(layers.Dense(units,
                                                   activation=self.activation_hidden))
        self.output_layer = layers.Dense(self.num_classes,
                                         activation=self.activation_out)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        outputs = self.output_layer(x)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'num_classes': self.num_outputs,
                      'units_values': self.units_values,
                      'activation_hidden': self.activation_hidden,
                      'activation_out': self.activation_out,
                      'normalization': self.normalization}
        config.update(new_config)
        return config
