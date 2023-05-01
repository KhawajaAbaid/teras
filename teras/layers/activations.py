import tensorflow as tf
from tensorflow.keras import layers

class GLU(layers.Layer):
    """Generalized linear unit nonlinear activation."""
    def __init__(self,
                units,
                **kwagrs):
        super().__init__(**kwagrs)
        self.units = units
    
    def call(self, inputs):
        return inputs[:, :self.units] * tf.nn.sigmoid(inputs[:, self.units:])


class GEGLU(layers.Layer):
    """GeGLU is an activation function which is a variant of GLU"""
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x, gates = tf.split(inputs,
                            num_or_size_splits=2,
                            axis=-1)
        return x * tf.nn.gelu(gates)