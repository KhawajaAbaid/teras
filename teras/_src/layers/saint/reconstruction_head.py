import keras
from keras import ops
from teras._src.layers.layer_list import LayerList
from teras._src.api_export import teras_export


@teras_export("teras.layers.SAINTReconstructionHead")
class SAINTReconstructionHead(keras.layers.Layer):
    """
    SAINT Reconstruction Head layer for `SAINTPretrainer`.
    For each feature in the dataset, it creates an MLP with a hidden
    layer and an output layer with dimensions equal to the cardinality of
    the feature in the case of a categorical features and 1 in the case
    of a continuous feature.

    Args:
        cardinalities: list or ndarray, a list or 1d-array of
            cardinalities of all the features in the dataset in the
            same order as the features' occurrence.
            For numerical features, use 0 as indicator at
            the corresponding index of the array.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
        embedding_dim: int, Dimensionality of embeddings being used in
            the model,
    """
    def __init__(self,
                 cardinalities: list,
                 embedding_dim: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim

        self.reconstruction_blocks = LayerList([
            LayerList([
                keras.layers.Dense(embedding_dim * 5,
                                   activation="relu",
                                   name=f"hidden_rb_{i}"),
                keras.layers.Dense((card + 1) if card == 0 else card)],
                sequential=True,
                name=f"reconstruction_block_{i}"
            )
            for i, card in enumerate(self.cardinalities)
        ],
            sequential=False,
            name="reconstruction_blocks"
        )

    def build(self, input_shape):
        self.reconstruction_blocks.build(input_shape)

    def call(self, inputs):
        reconstructed_features = self.reconstruction_blocks[0](
            inputs[:, 0, :]
        )
        for idx, layer in enumerate(self.reconstruction_blocks[1:],
                                    start=1):
            r_f = layer(inputs[:, idx, :])
            reconstructed_features = ops.concatenate(
                [reconstructed_features, r_f],
                axis=1
            )
        return reconstructed_features

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (len(self.cardinalities),)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cardinalities": self.cardinalities,
            "embedding_dim": self.embedding_dim,
        })
        return config
