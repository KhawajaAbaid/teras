import keras
from teras._src.api_export import teras_export


@teras_export("teras.layers.CTGANDiscriminatorLayer")
class CTGANDiscriminatorLayer(keras.layers.Layer):
    """
    Discriminator Layer based on the architecture proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    outputs = Dropout(LeakyReLU(Dense(inputs)))

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        dim: int, Dimensionality of the hidden layer. Default to 256.
        leaky_relu_alpha: float, Alpha value to use for leaky relu activation.
            Defaults to 0.2
        dropout_rate: float, Dropout rate to use in the `Dropout` layer,
            which is applied after hidden layer. Defaults to 0.
    """
    def __init__(self,
                 dim: int = 256,
                 leaky_relu_alpha: float = 0.2,
                 dropout_rate: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate

        self.dense = keras.layers.Dense(dim)
        self.leaky_relu = keras.layers.LeakyReLU(
            negative_slope=self.leaky_relu_alpha)
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    def build(self, input_shape):
        self.dense.build(input_shape)
        input_shape = self.dense.compute_output_shape(input_shape)
        self.leaky_relu.build(input_shape)
        self.dropout.build(input_shape)

    def call(self, inputs):
        out = self.dense(inputs)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'leaky_relu_alpha': self.leaky_relu_alpha,
            'dropout_rate': self.dropout_rate}
        )
        return config

