import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.ft_transformer")
class FTCLSToken(keras.layers.Layer):
    """
    CLS Token as proposed and implemented by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    It adds a BERT-like CLS token to the inputs, which is then used
    for the downstream task such as classification or regression.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        embedding_dim: ``int``, default 32,
            Embedding dimensions used to embed the numerical
            and categorical features.

        initialization: ``str``, default "normal",
            Initialization method to use for the weights.
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 initialization: str = "normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.initialization = initialization
        self.token = None

    def build(self, input_shape):
        initializer = self.initialization
        if isinstance(initializer, str):
            initializer = keras.initializers.get(initializer)
        self.token = self.add_weight(initializer=initializer,
                                     shape=(1, self.embedding_dim))

    def call(self, inputs):
        # We want to append the CLS token to the inputs
        # The expected input shape is, (batch_size, num_features, embedding_dim)
        # The self.token has the shape, (1, embedding_dim),
        # since batch_size can be dynamic, in that the last batch may have different shape
        # so we broadcast `self.token` to (`batch_size`, 1, `embedding_dim`), which in simple words
        # copies the `self.token` `batch_size` times along the first dimension.
        # And then we concatenate the `token_broadcasted` with `inputs` along the SECOND dimension,
        # i.e. the feature dimension, which gives us the final inputs of shape
        # (`batch_size`, `num_features + 1`, `embedding_dim`),
        # mind the plus one to `num_features`!
        token_broadcasted = tf.broadcast_to(self.token, shape=(tf.shape(inputs)[0], 1, self.embedding_dim))
        return tf.concat([inputs, token_broadcasted], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'embedding_dim': self.embedding_dim,
                       'initialization': self.initialization})
        return config
