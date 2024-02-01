import keras
from keras import ops
from teras.api_export import teras_export


@teras_export("teras.layers.SAINTEmbedding")
class SAINTEmbedding(keras.layers.Layer):
    def __init__(self,
                 embedding_dim: int,
                 cardinalities: list,
                 **kwargs):
        super().__init__(**kwargs)
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim

        self.embedding_layers = []
        for card in self.cardinalities:
            if card == 0:
                # it's continuous
                self.embedding_layers.append(
                    keras.layers.Dense(self.embedding_dim,
                                       activation="relu")
                )
            else:
                # it's categorical
                self.embedding_layers.append(
                    keras.layers.Embedding(
                        input_dim=card + 1,
                        output_dim=self.embedding_dim))

    def call(self, inputs):
        feature = ops.take(inputs, indices=[0], axis=1)
        # As much as I'd like to use the empty tensor, i can't because
        # every framework uses different methods for assigning values in
        # an **efficient** way. What's efficient in one framework isn't
        # efficient in the other. I ain't making a spaghetti. I'd rather do
        # it this way.
        embeddings = self.embedding_layers[0](feature)
        if len(ops.shape(embeddings)) == 2:
            embeddings = ops.expand_dims(embeddings, axis=1)
        for idx, embedding_layer in enumerate(self.embedding_layers[1:],
                                              start=1):
            feature = ops.take(inputs, indices=[idx], axis=1)
            feature_embeddings = embedding_layer(feature)
            if len(ops.shape(feature_embeddings)) == 2:
                feature_embeddings = ops.expand_dims(
                    feature_embeddings, axis=1)
            embeddings = ops.concatenate([embeddings, feature_embeddings],
                                         axis=1)
        return embeddings
