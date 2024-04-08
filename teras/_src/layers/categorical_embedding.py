import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.CategoricalEmbedding")
class CategoricalEmbedding(keras.layers.Layer):
    """
    Categorical Embedding layer that create trainable embeddings for
    categorical features values.

    Args:
        embedding_dim: int, dimensionality of the embeddings
        cardinalities: list or ndarray, a list or 1d-array of
            cardinalities of all the features in the dataset in the
            same order as the features' occurrence.
            For numerical features, use 0 as indicator at
            the corresponding index of the array.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
    """
    def __init__(self,
                 embedding_dim: int,
                 cardinalities: list,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.cardinalities = cardinalities
        self.embedding_layers = []
        self._categorical_idx = []
        for idx, card in enumerate(self.cardinalities):
            if card > 0:
                embedding = keras.layers.Embedding(
                    input_dim=card + 1,
                    output_dim=self.embedding_dim)
                self.embedding_layers.append(embedding)
                self._categorical_idx.append(idx)

    def call(self, inputs):
        idx = self._categorical_idx[0]
        feature = ops.expand_dims(inputs[:, idx], axis=1)
        categorical_embeddings = self.embedding_layers[0](feature)
        for feature_idx, embedding_layer in zip(self._categorical_idx[1:],
                                                self.embedding_layers[1:]):
            feature = ops.expand_dims(inputs[:, feature_idx], axis=1)
            f_e = embedding_layer(feature)
            # Concatenate feature embeddings along the feature axis
            categorical_embeddings = ops.concatenate(
                xs=[categorical_embeddings, f_e], axis=1)
        return categorical_embeddings

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (len(self._categorical_idx),
                                   self.embedding_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "cardinalities": self.cardinalities
        })
