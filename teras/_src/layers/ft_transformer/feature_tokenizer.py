import keras
import numpy as np
from teras._src.api_export import teras_export
from keras import ops
from keras.backend import floatx


@teras_export("teras.layers.FTTransformerFeatureTokenizer")
class FTTransformerFeatureTokenizer(keras.layers.Layer):
    """
    Feature Tokenizer layer based on FT-Transformer architecture
    proposed in the "Revisiting Deep Learning Models for Tabular Data"
    paper.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        cardinalities: list, a list cardinalities of all the features
            in the dataset in the same order as the features' occurrence.
            For numerical features, use any value <=0 as indicator at
            the corresponding index.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
        embedding_dim: int, dimensionality of the embeddings

    Shapes:
        Input Shape: `(batch_size, num_features)`
        Output Shape: `(batch_size, num_features, embedding_dim)`
    """
    def __init__(self,
                 cardinalities: list,
                 embedding_dim: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim
        self._cardinalities_arr = np.array(cardinalities)
        self._continuous_idx = np.flatnonzero(
            self._cardinalities_arr == 0)
        self._categorical_idx = np.flatnonzero(
            self._cardinalities_arr != 0)

        # we add an extra token for missing value
        num_special_tokens = 1
        self._total_tokens = sum(self.cardinalities) + num_special_tokens
        categorical_cardinalities = self._cardinalities_arr[self._categorical_idx]
        categorical_cardinalities = ops.pad(
            categorical_cardinalities,
            (1, 0),
            constant_values=num_special_tokens,
        )
        self._category_offset = ops.cumsum(categorical_cardinalities)[:-1]
        self._category_offset = ops.cast(self._category_offset, floatx())

    def build(self, input_shape=None):
        self.categorical_embeddings = self.add_weight(
            shape=(self._total_tokens, self.embedding_dim),
            initializer="random_normal",
            trainable=True,
        )
        num_continuous_features = len(self._continuous_idx)
        self.continuous_embeddings = self.add_weight(
            shape=(num_continuous_features,
                   self.embedding_dim),
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(len(self.cardinalities),),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        categorical = ops.take(inputs,
                               indices=self._categorical_idx,
                               axis=1)
        categorical += self._category_offset
        continuous = ops.take(inputs,
                              indices=self._continuous_idx,
                              axis=1)
        categorical = ops.cast(categorical, "int32")
        categorical = ops.take(self.categorical_embeddings,
                               indices=categorical,
                               axis=0)
        continuous = (ops.expand_dims(continuous, -1) *
                      ops.expand_dims(self.continuous_embeddings, 0))
        out = ops.concatenate([continuous, categorical], axis=1)
        out += ops.reshape(self.bias, (1, ops.shape(self.bias)[0], 1))
        return out

    def compute_output_shape(self, input_shape):
        return input_shape + (self.embedding_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cardinalities": self.cardinalities,
            "embedding_dim": self.embedding_dim
        })
        return config
