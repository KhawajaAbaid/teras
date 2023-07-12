import tensorflow as tf
from teras.layers.tabnet import (AttentiveTransformer,
                                 FeatureTransformer,
                                 Encoder,
                                 Decoder)


def test_tabnet_attentive_transformer_output_shape():
    attentive_transformer = AttentiveTransformer(num_features=16)
    inputs = tf.ones((128, 16))
    prior_scales = tf.ones(tf.shape(inputs))
    outputs = attentive_transformer(inputs, prior_scales)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16


def test_tabnet_feature_transformer_output_shape():
    feature_transformer = FeatureTransformer(units=64)
    inputs = tf.ones((128, 16))
    outputs = feature_transformer(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 64


def test_tabnet_encoder_output_shape():
    encoder = Encoder(feature_transformer_dim=64,
                      decision_step_output_dim=64,
                      num_features=16)
    inputs = tf.ones((128, 16))
    outputs = encoder(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 64


def test_tabnet_decoder_output_shape():
    encoder = Decoder(data_dim=16,
                      feature_transformer_dim=64,
                      decision_step_output_dim=64)
    inputs = tf.ones((128, 64))
    outputs = encoder(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16

