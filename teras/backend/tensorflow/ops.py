import tensorflow as tf


def norm(x, ord, axis, keepdims):
    return tf.norm(x, ord=ord, axis=axis, keepdims=keepdims)
