import tensorflow as tf


# TODO move this activations file to teras/Functional (the functional API, which is yet to be created!)
def glu(inputs, units):
    """Generalized linear unit nonlinear activation."""
    return inputs[:, :units] * tf.nn.sigmoid(inputs[:, units:])
