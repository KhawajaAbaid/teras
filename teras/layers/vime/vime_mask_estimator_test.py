import tensorflow as tf
from teras.layers.vime.vime_mask_estimator import VimeMaskEstimator


def test_vime_mask_estimator_valid_call():
    inputs = tf.ones((8, 5))
    mask_estimator = VimeMaskEstimator(units=16)
    outputs = mask_estimator(inputs)


def test_vime_mask_estimator_output_shape():
    inputs = tf.ones((8, 5))
    mask_estimator = VimeMaskEstimator(units=16)
    outputs = mask_estimator(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 16
