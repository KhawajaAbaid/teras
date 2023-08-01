from tensorflow import keras
from teras.activations import sparsemax


@keras.saving.register_keras_serializable("teras.layers.tabnet")
class TabNetAttentiveTransformer(keras.layers.Layer):
    """
    Attentive Transformer layer for mask generation
    as proposed by Sercan et al. in TabNet paper.
    It applies a ``Dense`` layer followed by a ``BatchNormalization`` layer
    followed by a ``sparsemax`` activation function.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: ``int``,
            Number of features in the input dataset.

        batch_momentum: ``float``, default 0.9,
            Momentum value to use for BatchNormalization layer.

        virtual_batch_size: ``int``, default 64,
            Batch size to use for ``virtual_batch_size`` parameter in ``BatchNormalization`` layer.

        relaxation_factor: ``float``, default 1.5,
            Relaxation factor that promotes the reuse of each feature at different
            decision steps. When it is 1, a feature is enforced to be used only at one
            decision step and as it increases, more flexibility is provided to use a
            feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for ``num_decision_steps`` favors for a larger ``relaxation_factor``.
    """
    def __init__(self,
                 data_dim: int,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 relaxation_factor: float = 1.5,
                 **kwargs):
        super().__init__(**kwargs)
        # We can't infer data dimensionality from build method because
        # the attentive transformer is applied after feature transformer
        # and hence received hidden representations of arbitrary dimensions
        self.data_dim = data_dim
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.relaxation_factor = relaxation_factor
        self.dense = keras.layers.Dense(self.data_dim, use_bias=False)
        self.batch_norm = keras.layers.BatchNormalization(momentum=batch_momentum,
                                                          virtual_batch_size=self.virtual_batch_size)

    def call(self, inputs, prior_scales=None):
        outputs = self.dense(inputs)
        outputs = self.batch_norm(outputs)
        outputs *= prior_scales
        outputs = sparsemax(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'data_dim': self.data_dim,
                       'batch_momentum': self.batch_momentum,
                       'virtual_batch_size': self.virtual_batch_size,
                       'relaxation_factor': self.relaxation_factor,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim, **config)
