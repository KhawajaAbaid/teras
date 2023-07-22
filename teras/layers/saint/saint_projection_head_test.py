import tensorflow as tf
from teras.layers.saint.saint_projection_head import SAINTProjectionHead


def test_saint_projection_head_output_shape():
    saint_transformer = SAINTProjectionHead(hidden_dim=16,
                                            output_dim=8)
    inputs = tf.ones((16, 10), dtype=tf.float32)
    outputs = saint_transformer(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 16
    assert tf.shape(outputs)[1] == 8
