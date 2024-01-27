import keras
from keras import ops
from teras.api_export import teras_export
from teras.utils.dtypes import ListOrArray


@teras_export("teras.layers.CategoricalExtraction")
class CategoricalExtraction(keras.layers.Layer):
    """
    CategoricalExtraction layer extracts categorical features from inputs
    as is. It helps us build functional model where inputs to the
    model contain both categorical and continuous features, but they
    must diverge into two different branches for separate processing.

    Args:
        categorical_idx: list or ndarray, list of indices of categorical
            features in the given dataset.
    """
    def __init__(self,
                 categorical_idx: ListOrArray,
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_idx = categorical_idx

    def call(self, inputs):
        categorical_features = ops.take(inputs,
                                        indices=self.categorical_idx,
                                        axis=1)
        return categorical_features
