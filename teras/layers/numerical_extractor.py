import keras
from keras import ops
from teras.api_export import teras_export


@teras_export("teras.layers.NumericalExtractor")
class NumericalExtractor(keras.layers.Layer):
    """
    NumericalExtractor layer extracts numerical features from inputs as is.
    It helps us build functional model where `CategoricalEmbedding`
    layer is used but no numerical embedding layer is applied.

    Args:
        numerical_idx: list or ndarray, list of indices of numerical
            features in the given dataset.
    """
    def __init__(self,
                 numerical_idx: list,
                 **kwargs):
        super().__init__(**kwargs)
        self.numerical_idx = numerical_idx

    def call(self, inputs):
        numerical_features = ops.take(inputs,
                                      indices=self.numerical_idx,
                                      axis=1)
        return numerical_features
