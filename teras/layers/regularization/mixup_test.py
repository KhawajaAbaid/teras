import tensorflow as tf
from teras.layers.regularization import MixUp


def test_mixup_valid_call():
    mixup = MixUp()
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = mixup(inputs)


def test_mixup_output_shape():
    mixup = MixUp()
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = mixup(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 4

