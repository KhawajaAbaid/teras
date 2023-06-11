import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from teras.layers.base.gan import HiddenBlock
from typing import List, Union, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


#  HEAR ME OUT! We NEED to expose this data_dim argument in the constructor
#   WHY?
#       Because the input to generator is almost always something differnet
#       from which we cannot infer input dimensions.
#       Exposing this argument will not only help users understand how things work
#       but also help take away much of the headache!


class Generator(keras.Model):
    """
    BaseGenerator is a Keras model that implement
    the common functionality that Generators in GAN based
    models need to implement, specifically in the context
    of GAN based architectures proposed for Tabular data.

    Args:
        data_dim: Dimensionality of the original dataset
        hidden_block: `keras.layers.Layer`, any subclass of `Layer` class
            that maybe be as simple as a `Dense` layer or may contain any
            complex logic with multiple sub layers such as `Dropout`,
            `Residual` etc.
        output_layer: `keras.layers.Layer`, any subclass of `Layer` class,
            such as `Dense` or a customized layer.
    """
    def __init__(self,
                 data_dim: int,
                 hidden_block: keras.layers.Layer = None,
                 output_layer: keras.layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.hidden_block = hidden_block
        self.output_layer = output_layer

        if self.hidden_block is None:
            units_values = [self.data_dim] * 2
            self.hidden_block = HiddenBlock(units_values=units_values)

        if self.output_layer is None:
            self.output_layer = layers.Dense(data_dim)

    def call(self, inputs):
        x = inputs
        x = self.hidden_block(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'data_dim': self.data_dim,
                       'hidden_block': self.hidden_layer,
                       'output_layer': self.output_layer,
                       })
        return config


class Discriminator(keras.Model):
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


class GAN(keras.Model):
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 num_discriminator_steps: int = 1,
                 data_dim: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.num_discriminator_steps = num_discriminator_steps

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    # Honestly speaking, there's no point in building a base GAN.
    # Why? because every GAN architecture is quite different in the
    # functionality that matters, i.e. the train step.
    # Why even bother building a base GAN, and complicating things
    # giving users extra burden of learning what the underlying structure
    # of base gan is, in addition to the keras.Model and then newcomers
    # may find it hard to differentiate what is what.
    # Also when to invoke what and what to pass to what. Sounds confusing, right?
    # For instance, i can't even have the base GAN implement the compile method,
    # because every generator and discriminator's loss is architecture dependent.
    # and then i won't be able to set the default values for them.
    # And if i do implement a compile method, with some arbitrary default values
    # that will just make things
    # so much more worse and complicated.
    # And not to mention, we're using Generator and Discriminator from base
    # directly, instead of subclassing them, but we will be subclassing base GAN
    # which just breaks the whole base thing and just causes more confusion.
    # Let's try to keep things simple and separate instead of trying to
    # fuse them together or trying to build them on another abstract component
    # or building block.
    # Our main building block
    # should just be the Layer,
    # and that's it.
    # No need at all to complicate things, for us and for the users.

