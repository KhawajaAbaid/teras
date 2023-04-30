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