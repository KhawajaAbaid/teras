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
        hidden_layer: `keras.layers.Layer`, any subclass of `Layer`
            class, such as `Dense` or customized layer such as a
            `ResidualBlock` etc.
            REMEMBER, you must pass the class and NOT instance of the class.
            It is used to construct the hidden block in the Generator.
            If `units_hidden` is passed but `hidden_layer` is None,
            then a `Dense` layer with `relu` activation is used.
        output_layer: `keras.layers.Layer`, any subclass of `Layer` class,
            such as `Dense` or a customized layer.
            You must pass the class and NOT instance of the class.
            If `units_out` is passed but `output_layer` is `None`,
            then a `Dense` layer with no activation is used.
        units_out: int, Number of units to use in output layer,
            or the dimensionality of the output.
            If None, no separate output dense layer will be added,
            and the output will have the dimensions of last hidden layer.
    """
    def __init__(self,
                 units_hidden: LIST_OR_TUPLE = None,
                 hidden_layer=None,
                 output_layer=None,
                 units_out: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units_hidden = units_hidden
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.units_out = units_out

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
                self.out = layers.Dense(self.units_out)
            else:
                self.out = self.output_layer(self.units_out)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        if self.out is not None:
            x = self.out(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'units_hidden': self.hidden_dims,
                       'hidden_layer': self.hidden_layer,
                       'output_layer': self.output_layer,
                       'units_out': self.units_out,
                       })
        return config


class BaseDiscriminator(keras.Model):
    """
    BaseDiscriminator is a Keras model that implements
    the common functionality that Discriminators in GAN based
    models need, specifically in the context
    of GAN based architectures proposed for Tabular data.

    Args:
        units_hidden: `list` | `tuple`, A list or tuple of
            units which determines the number and hidden
            dimensionality of the hidden layers to be created.
        hidden_layer: `keras.layers.Layer`, any subclass of `Layer`
            class, such as `Dense` or customized layer such as a
            `ResidualBlock` etc.
            REMEMBER, you must pass the class and NOT instance of the class.
            It is used to construct the hidden block in the Discriminator.
            If `units_hidden` is passed but `hidden_layer` is None,
            then a `Dense` layer with `relu` activation is used.
        output_layer: `keras.layers.Layer`, any subclass of `Layer` class,
            such as `Dense` or a customized layer.
            You must pass the class and NOT instance of the class.
            If `units_out` is passed but `output_layer` is `None`,
            then a `Dense` layer with no activation is used.
        units_out: int, Number of units to use in output layer,
            or the dimensionality of the output.
            If None, no separate output dense layer will be added,
            and the output will have the dimensions of last hidden layer.

    """
    def __init__(self,
                 units_hidden: LIST_OR_TUPLE = None,
                 hidden_layer=None,
                 output_layer=None,
                 units_out: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units_hidden = units_hidden
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.units_out = units_out

        self.hidden_block = None
        self.out = None

        if self.units_hidden is not None:
            self.hidden_block = models.Sequential(name="generator_hidden_block")
            for units in self.units_hidden:
                if self.hidden_layer is None:
                    self.hidden_block.add(layers.Dense(units, activation="relu"))
                else:
                    self.hidden_block.add(self.hidden_layer(units))
        if self.units_out is not None:
            if self.output_layer is None:
                self.out = layers.Dense(self.units_out)
            else:
                self.out = self.output_layer(self.units_out)

    def call(self, inputs):
        x = inputs
        if self.hidden_block is not None:
            x = self.hidden_block(x)
        if self.out is not None:
            x = self.out(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'units_hidden': self.hidden_dims,
                       'hidden_layer': self.hidden_layer,
                       'output_layer': self.output_layer,
                       'units_out': self.units_out,
                       })
        return config

