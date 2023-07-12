import tensorflow as tf
from teras.layers.oenf import PeriodicEmbedding


def test_oenf_periodic_embedding_output_shape():
    periodic_embedding = PeriodicEmbedding(num_features=16,
                                           embedding_dim=32)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = periodic_embedding(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16
    assert tf.shape(outputs)[2] == 32

