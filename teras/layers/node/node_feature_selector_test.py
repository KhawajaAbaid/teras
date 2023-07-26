from teras.layers.node.node_feature_selector import NodeFeatureSelector
import tensorflow as tf


def test_node_features_selector_valid_call_when_max_features_is_none():
    data_dim = 4
    intermediate_inputs = tf.ones((8, 16), dtype=tf.float32)
    feature_selector = NodeFeatureSelector(data_dim=data_dim,
                                           max_features=None)
    outputs = feature_selector(intermediate_inputs)


def test_node_features_selector_valid_call_when_max_features_is_not_none():
    data_dim = 4
    intermediate_inputs = tf.ones((8, 16), dtype=tf.float32)
    feature_selector = NodeFeatureSelector(data_dim=data_dim,
                                           max_features=8)
    outputs = feature_selector(intermediate_inputs)


def test_node_features_selector_output_shape_when_max_features_is_none():
    data_dim = 4
    intermediate_inputs = tf.ones((8, 16), dtype=tf.float32)
    feature_selector = NodeFeatureSelector(data_dim=data_dim,
                                           max_features=None)
    outputs = feature_selector(intermediate_inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    # It must be equal to the dimensions of intermediate dimensions i.e. 16
    assert tf.shape(outputs)[1] == 16


def test_node_features_selector_output_shape_when_max_features_is_not_none():
    data_dim = 4
    intermediate_inputs = tf.ones((8, 16), dtype=tf.float32)
    feature_selector = NodeFeatureSelector(data_dim=data_dim,
                                           max_features=8)
    outputs = feature_selector(intermediate_inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    # It must have data_dim + (min(intermediate_inputs.shape[1], max_features) - data_dimensions) dimensions
    assert tf.shape(outputs)[1] == 8
