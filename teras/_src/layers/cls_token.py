import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.CLSToken")
class CLSToken(keras.layers.Layer):
    """
    CLS Token layer that makes it possible to append CLS token embedding
    to the input embeddings in the sequential or functional models.

    The idea of CLS token was introduced in the "BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding" paper.

    Reference(s):
        https://arxiv.org/abs/1810.04805

    Args:
        embedding_dim: int, dimensionality of the input embeddings

    Shapes:
        Input Shape: `(batch_size, num_features, embedding_dim)`
        Output Shape: `(batch_size, num_features + 1, embedding_dim)`
    """
    def __init__(self,
                 embedding_dim: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(
            shape=(1, self.embedding_dim),
            initializer="random_normal",
        )

    def call(self, inputs):
        # TODO Remove the call to `convert_to_tensor` as soon as Keras
        #  fixes `broadcast_to` method for JAX backend
        token_broadcasted = ops.broadcast_to(
            ops.convert_to_tensor(self.cls_token),
            shape=(ops.shape(inputs)[0], *ops.shape(self.cls_token)))
        return ops.concatenate([token_broadcasted, inputs],
                               axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim
        })
        return config
