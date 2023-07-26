import tensorflow as tf
from teras.layers.dnfnet.dnnf import DNNF


def test_dnnf_valid_call():
    dnnf = DNNF(num_formulas=16)
    inputs = tf.ones((10, 8), dtype=tf.float32)
    outputs = dnnf(inputs)


def test_dnnf_output_shape():
    dnnf = DNNF(num_formulas=16)
    inputs = tf.ones((8, 10), dtype=tf.float32)
    outputs = dnnf(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 16
