from teras.layers.activations import GLU, GEGLU, GumbelSoftmax
import tensorflow as tf


def test_glu_output_shape():
    glu = GLU(units=8)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = glu(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 8


def test_geglu_output_shape():
    geglu = GEGLU()
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = geglu(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 8


def test_gumble_softmax_output_shape():
    gumble_softmax = GumbelSoftmax()
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = gumble_softmax(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16



