from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from typing import List, Union, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class HiddenBlock(keras.layers.Layer):
    def __init__(self,
                 units_values: LIST_OR_TUPLE,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units_values = units_values
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.block = models.Sequential()
        for units in self.units_values:
            self.block.add(layers.Dense(units,
                                        activation=self.activation,
                                        use_bias=self.use_bias,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer=self.bias_initializer,
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        activity_regularizer=self.activity_regularizer,
                                        kernel_constraint=self.kernel_constraint,
                                        bias_constraint=self.bias_constraint))

    def call(self, inputs):
        return self.block(inputs)
