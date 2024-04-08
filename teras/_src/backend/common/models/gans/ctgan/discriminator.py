import keras
from keras import random, ops
from teras._src.layers.ctgan.discriminator_layer import CTGANDiscriminatorLayer
from teras._src.typing import IntegerSequence
from teras._src.layers.layer_list import LayerList


class BaseCTGANDiscriminator(keras.Model):
    """
    Base CTGANDiscriminator class.
    """
    def __init__(self,
                 hidden_dims: IntegerSequence = (256, 256),
                 packing_degree: int = 8,
                 gradient_penalty_lambda: float = 10.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.packing_degree = packing_degree
        self.gradient_penalty_lambda = gradient_penalty_lambda
        self.seed = seed
        self.seed_gen = random.SeedGenerator(self.seed)

        self.hidden_block = []
        for dim in self.hidden_dims:
            self.hidden_block.append(CTGANDiscriminatorLayer(dim))
        self.hidden_block = LayerList(
            self.hidden_block,
            sequential=True,
            name="discriminator_hidden_block"
        )
        self.output_layer = keras.layers.Dense(
            1,
            name="discriminator_output_layer")

    def build(self, input_shape):
        batch_size, input_dim = input_shape
        input_shape = (batch_size // self.packing_degree,
                       input_dim * self.packing_degree)
        self.hidden_block.build(input_shape)
        input_shape = self.hidden_block.compute_output_shape(input_shape)
        self.output_layer.build(input_shape)

    def call(self, inputs):
        inputs_dim = ops.shape(inputs)[1]
        inputs = ops.reshape(inputs,
                             newshape=(-1, self.packing_degree * inputs_dim))
        outputs = self.hidden_block(inputs)
        outputs = self.output_layer(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        new_config = {
            'hidden_dims': self.hidden_dims,
            'packing_degree': self.packing_degree,
            'gradient_penalty_lambda': self.gradient_penalty_lambda,
            'seed': self.seed,
        }
        config.update(new_config)
        return config

