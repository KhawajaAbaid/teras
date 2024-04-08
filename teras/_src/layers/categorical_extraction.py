import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.CategoricalExtraction")
class CategoricalExtraction(keras.layers.Layer):
    """
    CategoricalExtraction layer extracts categorical features from inputs
    as is. It helps us build functional model where inputs to the
    model contain both categorical and continuous features, but they
    must diverge into two different branches for separate processing.

    Args:
        categorical_idx: list, list of indices of categorical features in the
            given dataset.
    """
    def __init__(self,
                 categorical_idx: list,
                 **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.categorical_idx = categorical_idx

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = len(self.categorical_idx)
        return tuple(output_shape)

    def call(self, inputs):
        categorical_features = ops.take(inputs,
                                        indices=self.categorical_idx,
                                        axis=1)
        return categorical_features

    def get_config(self):
        config = super().get_config()
        config.update({
            "categorical_idx": self.categorical_idx
        })
