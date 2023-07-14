from teras.layerflow.models.node import NODERegressor, NODEClassifier
import tensorflow as tf


def test_node_classifier_works():
    data = tf.ones(shape=(16, 10))
    node_classifier = NODEClassifier()
    node_classifier(data)


def test_node_regressor_works():
    data = tf.ones(shape=(16, 10))
    node_regressor = NODERegressor()
    node_regressor(data)
