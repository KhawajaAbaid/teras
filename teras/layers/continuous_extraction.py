import keras
from keras import ops
from teras.api_export import teras_export
from teras.utils.dtypes import ListOrArray


@teras_export("teras.layers.ContinuousExtractor")
class ContinuousExtractor(keras.layers.Layer):
    """
    ContinuousExtractor layer extracts continuous features from inputs 
    as is. It helps us build functional model where inputs must diverge
    into categorical and continuous parts for separate  processing.

    Args:
        continuous_idx: list or ndarray, list of indices of continuous
            features in the given dataset.
    """
    def __init__(self,
                 continuous_idx: ListOrArray,
                 **kwargs):
        super().__init__(**kwargs)
        self.continuous_idx = continuous_idx

    def call(self, inputs):
        continuous_features = ops.take(inputs,
                                       indices=self.continuous_idx,
                                       axis=1)
        return continuous_features
