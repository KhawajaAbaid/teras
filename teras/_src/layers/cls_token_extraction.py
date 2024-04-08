import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.CLSTokenExtraction")
class CLSTokenExtraction(keras.layers.Layer):
    """
    Extracts CLS Token embeddings.
    Main purpose is to make it easy to build sequential or functional
    models.

    Args:
        axis: int, defaults to 1.

    Shapes:
        Input Shape: `(batch_size, num_features, embedding_dim)`
        Output Shape: `(batch_size, 1, emebdding_dim)`
    """
    def __init__(self,
                 axis: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return ops.take(inputs, indices=[0], axis=self.axis)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config
