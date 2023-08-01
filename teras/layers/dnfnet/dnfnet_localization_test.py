import tensorflow as tf
from teras.layers.dnfnet.dnfnet_localization import DNFNetLocalization


def test_dnfnet_localitzation_valid_call():
    localization = DNFNetLocalization(num_formulas=16)
    inputs = tf.ones((8, 10), dtype=tf.float32)
    outputs = localization(inputs)


def test_dnfnet_localitzation_output_shape():
    localization = DNFNetLocalization(num_formulas=16)
    inputs = tf.ones((8, 10), dtype=tf.float32)
    outputs = localization(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 16
