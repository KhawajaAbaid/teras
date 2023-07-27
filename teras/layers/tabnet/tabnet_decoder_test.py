import tensorflow as tf
from teras.layers.tabnet.tabnet_decoder import TabNetDecoder


def test_tabnet_decoder_valid_call():
    decoder = TabNetDecoder(data_dim=5,
                            feature_transformer_dim=32,
                            decision_step_output_dim=32)
    inputs = tf.random.normal((8, 16))
    outputs = decoder(inputs)


def test_tabnet_decoder_output_shape():
    decoder = TabNetDecoder(data_dim=5,
                            feature_transformer_dim=32,
                            decision_step_output_dim=32)
    inputs = tf.random.normal((8, 16))
    outputs = decoder(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 5
