import tensorflow as tf
from teras.layers.tabnet.tabnet_feature_transformer import TabNetFeatureTransformer


def test_tabnet_feature_transformer_valid_call():
    TabNetFeatureTransformer.reset_shared_layers()
    feature_transformer = TabNetFeatureTransformer(units=32)
    inputs = tf.ones((8, 10))
    outputs = feature_transformer(inputs)


def test_tabnet_feature_transformer_output_shape():
    TabNetFeatureTransformer.reset_shared_layers()
    feature_transformer = TabNetFeatureTransformer(units=32)
    inputs = tf.ones((8, 10))
    outputs = feature_transformer(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 32
