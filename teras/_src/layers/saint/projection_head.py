import keras
from teras._src.api_export import teras_export


@teras_export("teras.layers.SAINTProjectionHead")
class SAINTProjectionHead(keras.layers.Layer):
    """
    Projection Head layer that is used in the contrastive learning phase of
    the `SAINTPretrainer` to project embeddings to a lower dimension.
    According to the SAINT paper,
    "The use of a projection head to reduce dimensionality before computing
    contrastive loss is common in vision and indeed also improves results
    on tabular data."

    Reference(s):
    https://arxiv.org/abs/2106.01342

    Args:
        hidden_dim: int, Dimensionality of the hidden layer.
            In the official implementation, it is computed as follows,
            `hidden_dim = 6 * embedding_dim * number_of_features // 5`
        output_dim: int, Dimensionality of the output layer.
            In the official implementation, it is computed as follows,
            `output_dim = embedding_dim * number_of_features // 2`
        hidden_activation: Activation function to use in the hidden layer.
            Defaults to "relu".
    """
    def __init__(self,
                 hidden_dim: int,
                 output_dim: int,
                 hidden_activation="relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim

        self.hidden_block = keras.layers.Dense(
            units=hidden_dim,
            activation=hidden_activation,
            name="projection_head_hidden"
        )
        self.output_layer = keras.layers.Dense(
            units=output_dim,
            name="projection_head_output"
        )

    def build(self, input_shape):
        self.hidden_block.build(input_shape)
        input_shape = self.hidden_block.compute_output_shape(input_shape)
        self.output_layer.build(input_shape)

    def call(self, inputs):
        x = self.hidden_block(inputs)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)

    def get_config(self):
        config = {
            "name": self.name,
            "trainable": self.trainable,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "hidden_activation": self.hidden_activation,
        }
        return config
