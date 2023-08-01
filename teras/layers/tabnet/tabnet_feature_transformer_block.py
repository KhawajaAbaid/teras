import tensorflow as tf
from tensorflow import keras
from teras.activations import glu


@keras.saving.register_keras_serializable("teras.layers.tabnet")
class TabNetFeatureTransformerBlock(keras.layers.Layer):
    """
    Feature Transformer block layer is used in constructing the FeatureTransformer
    as proposed by Sercan et al. in TabNet paper.
    It applies a Dense layer followed by a BatchNormalization layer
    followed by a glu activation function.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: ``int``, default 32,
            the dimensionality of the hidden representation in feature transformation block.
            Each layer first maps the representation to a ``2 * feature_transformer_dim``
            output and half of it is used to determine the
            non-linearity of the ``glu`` activation where the other half is used as an
            input to ``glu``, and eventually ``feature_transformer_dim`` output is
            transferred to the next layer.

        batch_momentum: ``float``, default 0.9,
            Momentum value to use for ``BatchNormalization`` layer.

        virtual_batch_size: ``int``, default 64,
            Batch size to use for ``virtual_batch_size`` parameter in ``BatchNormalization`` layer.
            This is typically much smaller than the ``batch_size`` used for training.
            And most importantly, the ``batch_size`` must always be divisible by the ``virtual_batch_size``.

        residual_normalization_factor: ``float``, default 0.5,
            In the feature transformer, except for the layer, every other layer utilizes normalized residuals,
            where ``residual_normalization_factor`` determines the scale of normalization.

        use_residual_normalization: ``bool``, default True,
            Whether to use residual normalization.
            According to the default architecture, every layer uses residual normalization
            EXCEPT for the very first layer.
    """
    def __init__(self,
                 units: int = 32,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 use_residual_normalization: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.use_residual_normalization = use_residual_normalization
        self.residual_normalization_factor = residual_normalization_factor
        self.dense = keras.layers.Dense(self.units * 2, use_bias=False)
        self.norm = keras.layers.BatchNormalization(momentum=self.batch_momentum,
                                                    virtual_batch_size=virtual_batch_size)
        self.add = keras.layers.Add()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = glu(x)
        if self.use_residual_normalization:
            x = self.add([x, inputs]) * tf.math.sqrt(self.residual_normalization_factor)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units,
                       'batch_momentum': self.batch_momentum,
                       'virtual_batch_size': self.virtual_batch_size,
                       'residual_normalization_factor': self.residual_normalization_factor,
                       'use_residual_normalization': self.use_residual_normalization,
                       })
        return config
