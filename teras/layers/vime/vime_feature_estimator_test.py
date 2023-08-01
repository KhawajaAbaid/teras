import tensorflow as tf
from teras.layers.vime.vime_feature_estimator import VimeFeatureEstimator


def test_vime_feature_estimator_valid_call():
    inputs = tf.ones((8, 5))
    feature_estimator = VimeFeatureEstimator(units=16)
    outputs = feature_estimator(inputs)


def test_vime_feature_estimator_output_shape():
    inputs = tf.ones((8, 5))
    feature_estimator = VimeFeatureEstimator(units=16)
    outputs = feature_estimator(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 16
