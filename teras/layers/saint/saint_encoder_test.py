import tensorflow as tf
from teras.layers.saint.saint_encoder import SAINTEncoder


def test_saint_encoder_output_shape():
    encoder = SAINTEncoder(data_dim=10,
                           embedding_dim=32)
    inputs = tf.ones((16, 10, 32), dtype=tf.float32)
    outputs = encoder(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 16
    assert tf.shape(outputs)[1] == 10
    assert tf.shape(outputs)[2] == 32
