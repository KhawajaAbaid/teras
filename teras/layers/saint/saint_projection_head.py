from tensorflow import keras
from teras.layerflow.layers.saint.saint_projection_head import SAINTProjectionHead as _SAINTProjectionHeadLF


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class SAINTProjectionHead(_SAINTProjectionHeadLF):
    """
    ProjectionHead layer that is used in the contrastive learning phase of
    the SAINTPretrainer to project embeddings to a lower dimension.
    According to the SAINT paper,
    "The use of a projection head to reduce dimensionality before computing
    contrastive loss is common in vision and indeed also improves results
    on tabular data."

    Reference(s):
    https://arxiv.org/abs/2106.01342

    Args:
        hidden_dim: ``int``, default 64,
            Dimensionality of the hidden layer.
            In the official implementation, it is computed as follows,
            `hidden_dim = 6 * embedding_dim * number_of_features // 5`

        hidden_activation: default "relu":
            Activation function to use in the hidden layer.

        output_dim: ``int``, default 32,
            Dimensionality of the output layer.
            In the official implementation, it is computed as follows,
            `output_dim = embedding_dim * number_of_features // 5`
    """
    def __init__(self,
                 hidden_dim: int = 64,
                 hidden_activation="relu",
                 output_dim: int = 32,
                 **kwargs):
        hidden_block = keras.layers.Dense(units=hidden_dim,
                                          activation=hidden_activation)
        output_layer = keras.layers.Dense(units=output_dim)

        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'hidden_dim': self.hidden_dim,
                  'hidden_activation': self.hidden_activation,
                  'output_dim': self.output_dim}
        return config

    # need to override the from_config since the layerflow version is parent
    # and it tries to pop out layers to deserialize which in this default
    # api case don't exist in the configuration and hence causes KeyError
    @classmethod
    def from_config(cls, config):
        return cls(**config)
