import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layers.vime")
class VimeMaskGenerationAndCorruption(keras.layers.Layer):
    """
    A stateless layer that generates masks and corrupted samples
    By implementing this functionality in a layer's call method, we can use this layer in a sequential or functional
    manner and this way the layer and hence the generator functions it calls will have access to the batch size dimension
    because we'll precede this layer with an input layer that specifies the batch dimension and input shape in general
    and pack these both in a sequential model or plug them together using functional API

    Args:
        p_m: ``float``, default 0.3,
            Corruption probability. (Note: Don't do it in real life!)
    """
    def __init__(self,
                 p_m: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.p_m = p_m

    def call(self, inputs):
        """inputs: Batch of unlabeled X"""

        # Generate Mask Vector
        mask = tf.random.stateless_binomial(shape=tf.shape(inputs),
                                            seed=(0, 0),
                                            counts=1,
                                            probs=self.p_m,
                                            output_dtype=tf.float32)

        # Generate corrupted samples
        num_samples = tf.shape(inputs)[0]
        dim = tf.shape(inputs)[1]

        x_bar = tf.TensorArray(size=dim, dtype=tf.float32)
        for i in range(dim):
            idx = tf.random.shuffle(tf.range(num_samples))
            x_bar = x_bar.write(i, tf.gather(inputs[:, i], idx))
        x_bar = tf.transpose(x_bar.stack())

        # Corrupt Samples
        x_tilde = inputs * (1 - mask) + x_bar * mask
        return x_tilde

    def get_config(self):
        config = super().get_config()
        config.update({'p_m': self.p_m,
                       })
        return config
