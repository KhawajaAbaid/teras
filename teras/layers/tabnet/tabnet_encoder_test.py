import tensorflow as tf
from teras.layers.tabnet.tabnet_encoder import TabNetEncoder


def test_tabnet_encoder_valid_call():
    encoder = TabNetEncoder(data_dim=10,
                            feature_transformer_dim=32,
                            decision_step_output_dim=32)
    inputs = tf.ones((8, 10))
    outputs = encoder(inputs)


def test_tabnet_encoder_output_shape():
    encoder = TabNetEncoder(data_dim=10,
                            feature_transformer_dim=32,
                            decision_step_output_dim=32)
    inputs = tf.ones((8, 10))
    outputs = encoder(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 32
