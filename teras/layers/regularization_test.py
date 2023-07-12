import tensorflow as tf
from teras.layers.regularization import MixUp, CutMix


def test_mixup_output_shape():
    mixup = MixUp()
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = mixup(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16


def test_cutmix_output_shape():
    cutmix = CutMix()
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = cutmix(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16
