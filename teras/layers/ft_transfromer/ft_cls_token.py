import tensorflow as tf
from tensorflow import keras


# TODO rework this layer -- ideally making it simple!
class FTCLSToken(keras.layers.Layer):
    """
    CLS Token as proposed and implemented by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

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

    def build(self, input_shape):
        initializer = self.initialization
        if isinstance(initializer, str):
            initializer = keras.initializers.get(initializer)
        self.weight = self.add_weight(initializer=initializer,
                                      shape=(self.embedding_dim,))

    def expand(self, *leading_dimensions: int) -> tf.Tensor:
        if not leading_dimensions:
            return self.weight
        return tf.broadcast_to(tf.expand_dims(self.weight, axis=0), (*leading_dimensions, self.weight.shape[0]))

    def call(self, inputs):
        return tf.concat([inputs, self.expand(tf.shape(inputs)[0], 1)], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'embedding_dim': self.embedding_dim,
                      'initialization': self.initialization})
        return config
