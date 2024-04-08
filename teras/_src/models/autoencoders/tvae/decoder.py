import keras

from teras._src.layers.layer_list import LayerList
from teras._src.typing import IntegerSequence
from teras._src.api_export import teras_export


@teras_export("teras.models.TVAEDecoder")
class TVAEDecoder(keras.Model):
    """
    Encoder for the TVAE model as proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: int, Dimensionality of the input dataset.
        decompression_dims: A sequence of integers. For each value in
            the sequence, a dense layer of that dimensionality is added to
            construct a decompression block.
    """
    def __init__(self,
                 data_dim: int,
                 decompression_dims: IntegerSequence = (128, 128),
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(decompression_dims, (list, tuple)):
            raise ValueError(
                f"`decompression_dims` must be a sequence of integers. "
                f"Received: {decompression_dims}")

        self.data_dim = data_dim
        self.decompression_dims = decompression_dims

        self.compression_block = []
        for i, units in enumerate(self.decompression_dims, start=1):
            self.compression_block.append(
                keras.layers.Dense(units=units,
                                   activation="relu",
                                   name=f"decompression_layer_{i}"))
        self.decompression_block = LayerList(
            self.compression_block,
            sequential=True,
            name="tvae_encoder_compression_block"
        )
        self.projection_layer = keras.layers.Dense(self.data_dim,
                                                   name="projection_layer")
        self.sigmas = self.add_weight(shape=(self.data_dim,),
                                      initializer="ones", trainable=True,
                                      name="sigmas") * 0.1

    def build(self, input_shape):
        self.decompression_block.build(input_shape)
        input_shape = self.decompression_block.compute_output_shape(input_shape)
        self.projection_layer.build(input_shape)

    def call(self, inputs):
        x_generated = self.projection_layer(self.decompression_block(inputs))
        return x_generated, self.sigmas

    def predict_step(self, z):
        generated_samples, _ = self(z)
        return generated_samples

    def compute_output_shape(self, input_shape):
        batch_size, input_dim = input_shape
        return ((batch_size, self.data_dim),
                (self.data_dim,))

    def get_config(self):
        config = super().get_config()
        config.update({
            'data_dim': self.data_dim,
            'decompression_dims': self.decompression_dims
        })
        return config

