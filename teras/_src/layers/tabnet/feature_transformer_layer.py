import keras
from teras._src.activations import glu
from teras._src.api_export import teras_export


@teras_export("teras.layers.TabNetFeatureTransformerLayer")
class TabNetFeatureTransformerLayer(keras.layers.Layer):
    """
    TabNetFeatureTransformerLayer layer that serves as the building block
    for the `TabNetFeatureTransformer` layer which is proposed by Arik
    et al. in the "TabNet: Attentive Interpretable Tabular Learning" paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        dim: int, the dense layer first maps the inputs to dim * 2
            dimension hidden representations and later the glu activation
            maps the hidden representations to `dim`-dimensions.
        batch_momentum: float, batch momentum
    """
    def __init__(self,
                 dim: int,
                 batch_momentum: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.batch_momentum = batch_momentum

        self.dense = keras.layers.Dense(self.dim * 2,
                                        use_bias=False)
        self.batch_norm = keras.layers.BatchNormalization(
            momentum=self.batch_momentum)

    def build(self, input_shape):
        self.dense.build(input_shape)
        input_shape = self.dense.compute_output_shape(input_shape)
        self.batch_norm.build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        return glu(x)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "batch_momentum": self.batch_momentum,
        })
        return config
