import keras

from teras._src.layers.layer_list import LayerList
from teras._src.typing import IntegerSequence
from teras._src.api_export import teras_export


@teras_export("teras.models.TVAEEncoder")
class TVAEEncoder(keras.Model):
    """
    Encoder for the TVAE model as proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        latent_dim: int, Dimensionality of the learned latent space.
            Defaults to 128.
        compression_dims: Sequence, A sequence of integers. For each value in 
            the sequence, a dense layer of that dimensions is added to 
            construct a compression block.
            Defaults to (128, 128).
    """
    def __init__(self,
                 latent_dim: int = 128,
                 compression_dims: IntegerSequence = (128, 128),
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(compression_dims, (list, tuple)):
            raise ValueError(
                f"`compression_dims` must be a sequence of integers."
                f"Received: {compression_dims}")

        self.latent_dim = latent_dim
        self.compression_dims = compression_dims

        self.compression_block = []
        for i, units in enumerate(self.compression_dims, start=1):
            self.compression_block.append(
                keras.layers.Dense(units=units,
                                   activation="relu",
                                   name=f"compression_layer_{i}"))
        self.compression_block = LayerList(
            self.compression_block,
            sequential=True,
            name="tvae_encoder_compression_block"
        )
        self.dense_mean = keras.layers.Dense(self.latent_dim,
                                             name="mean")
        self.dense_log_var = keras.layers.Dense(self.latent_dim,
                                                name="log_var")

    def build(self, input_shape):
        self.compression_block.build(input_shape)
        input_shape = self.compression_block.compute_output_shape(input_shape)
        self.dense_mean.build(input_shape)
        self.dense_log_var.build(input_shape)

    def call(self, inputs):
        h = self.compression_block(inputs)
        mean = self.dense_mean(h)
        log_var = self.dense_log_var(h)
        return mean, log_var

    def compute_output_shape(self, input_shape):
        batch_size, dims = input_shape
        return ((batch_size, self.latent_dim),
                (batch_size, self.latent_dim))

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'compression_dims': self.compression_dims
        })
        return config

