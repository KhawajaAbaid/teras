import tensorflow as tf
from teras.layers.tabnet.tabnet_attentive_transformer import TabNetAttentiveTransformer


def test_tabnet_attentive_transformer_valid_call():
    attentive_transformer = TabNetAttentiveTransformer(data_dim=5)
    inputs = tf.ones((8, 5))
    prior_scales = tf.ones(tf.shape(inputs))
    outputs = attentive_transformer(inputs, prior_scales)


def test_tabnet_attentive_transformer_output_shape():
    attentive_transformer = TabNetAttentiveTransformer(data_dim=5)
    inputs = tf.ones((8, 5))
    prior_scales = tf.ones(tf.shape(inputs))
    outputs = attentive_transformer(inputs, prior_scales)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 5

