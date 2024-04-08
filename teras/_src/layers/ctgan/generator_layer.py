import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.CTGANGeneratorLayer")
class CTGANGeneratorLayer(keras.layers.Layer):
    """
    Residual Block for Generator as used by the authors of CTGAN
    proposed in the paper Modeling Tabular data using Conditional GAN.

    `outputs = Concat([ReLU(BatchNorm(Dense(inputs))), inputs])`

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        dim: int, Dimensionality of the hidden layer.
            Defaults to 256.
    """
    def __init__(self,
                 dim: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dense = keras.layers.Dense(self.dim)
        self.batch_norm = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

    def build(self, input_shape):
        self.dense.build(input_shape)
        input_shape = self.dense.compute_output_shape(input_shape)
        self.batch_norm.build(input_shape)
        self.relu.build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        out = ops.concatenate([x, inputs], axis=1)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + self.dim,)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config
