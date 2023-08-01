import tensorflow as tf
from teras.layers.vime.vime_encoder import VimeEncoder


def test_vime_feature_estimator_valid_call():
    inputs = tf.ones((8, 5))
    encoder = VimeEncoder(units=16)
    outputs = encoder(inputs)


def test_vime_feature_estimator_output_shape():
    inputs = tf.ones((8, 5))
    encoder = VimeEncoder(units=16)
    outputs = encoder(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 16
