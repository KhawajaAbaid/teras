import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.ContinuousExtraction")
class ContinuousExtraction(keras.layers.Layer):
    """
    ContinuousExtraction layer extracts continuous features from inputs
    as is. It helps us build functional model where inputs to the
    model contain both categorical and continuous features, but they
    must diverge into two different branches for separate processing.

    Args:
        continuous_idx: list, a list of indices of continuous
            features in the given dataset.
    """
    def __init__(self,
                 continuous_idx: list,
                 **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.continuous_idx = continuous_idx

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = len(self.continuous_idx)
        return tuple(output_shape)

    def call(self, inputs):
        continuous_features = ops.take(inputs,
                                       indices=self.continuous_idx,
                                       axis=1)
        return continuous_features

    def get_config(self):
        config = super().get_config()
        config.update({
            "continuous_idx": self.continuous_idx
        })
