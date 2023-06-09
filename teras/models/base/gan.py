import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from typing import List, Union, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class BaseGenerator(keras.Model):
    """
    GeneratorBackbone is a Keras model that implement
    the common functionality that Generators in GAN based
    models need to implement, specifically in the context
    of GAN based architectures proposed for Tabular data.

    Args:
        units_hidden: `list` | `tuple`, A list or tuple of
            units which determines the number and hidden
            dimensionality of the building blocks to be created.
        hidden_layer: Either `layers.Dense` class
            or a `subclass of layers.Dense`
            You must pass the class and NOT instance of the class.
            Used to construct the hidden block in the Generator.
            If units_hidden are passed but hidden_layer is None,
            then a `Dense` layer with `relu` activation is used.
        units_out: int, Number of units to use in output dense layer,
            or the dimensionality of the output.
            If None, no separate output dense layer will be added,
            and the output will have the dimensions of last hidden layer.
    """
    def __init__(self,
                 units_hidden: LIST_OR_TUPLE = None,
                 hidden_layer=None,
                 units_out: int = None,
                 output_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units_hidden = units_hidden
        self.hidden_layer = hidden_layer
        self.units_out = units_out
        self.output_layer = output_layer

        self.hidden_block = None
        self.dense_out = None

        if self.units_hidden is not None:
            self.hidden_block = models.Sequential(name="generator_hidden_block")
            for units in self.units_hidden:
                if self.hidden_layer is None:
                    self.hidden_block.add(layers.Dense(units, activation="relu"))
                else:
                    self.hidden_block.add(self.hidden_layer(units))
        if self.units_out is not None:
            if self.output_layer is None:
                self.dense_out = layers.Dense(self.units_out)
            else:
                self.dense_out = self.output_layer(self.units_out)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        if self.dense_out is not None:
            x = self.dense_out(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'units_hidden': self.hidden_dims,
                       'hidden_layer': self.hidden_layer,
                       'units_out': self.units_out,
                       'output_layer': self.output_layer,
                       })
        return config
