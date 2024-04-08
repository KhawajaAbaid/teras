import keras
from teras._src.activations import sparsemax
from teras._src.api_export import teras_export


@teras_export("teras.layers.TabNetAttentiveTransformer")
class TabNetAttentiveTransformer(keras.layers.Layer):
    """
    TabNetAttentiveTransformer layer proposed by Arik et al. in the
    "TabNet: Attentive Interpretable Tabular Learning" paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: int, dimensionality of the dataset
        batch_momentum: float, batch momentum
    """
    def __init__(self,
                 data_dim: int,
                 batch_momentum: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.batch_momentum = batch_momentum

        self.dense = keras.layers.Dense(data_dim,
                                        use_bias=False)
        self.batch_norm = keras.layers.BatchNormalization(
            momentum=self.batch_momentum)

    def build(self, input_shape):
        self.dense.build(input_shape)
        input_shape = self.dense.compute_output_shape(input_shape)
        self.batch_norm.build(input_shape)
        self.built = True

    def call(self, inputs, prior_scales):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = x * prior_scales
        x = sparsemax(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.data_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "data_dim": self.data_dim,
            "batch_momentum": self.batch_momentum
        })
    