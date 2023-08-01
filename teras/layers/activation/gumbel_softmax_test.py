from teras.layers.activation import GumbelSoftmax
import tensorflow as tf


def test_gumble_softmax_valid_call():
    gumble_softmax = GumbelSoftmax()
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = gumble_softmax(inputs)


def test_gumble_softmax_output_shape():
    gumble_softmax = GumbelSoftmax()
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = gumble_softmax(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 4



