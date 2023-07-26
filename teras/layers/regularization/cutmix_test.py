import tensorflow as tf
from teras.layers.regularization import CutMix


def test_cutmix_valid_call():
    cutmix = CutMix()
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = cutmix(inputs)


def test_cutmix_output_shape():
    cutmix = CutMix()
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = cutmix(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 4
