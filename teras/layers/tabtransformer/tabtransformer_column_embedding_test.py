import tensorflow as tf
from teras.layers.tabtransformer.tabtransformer_column_embedding import TabTransformerColumnEmbedding


def test_tabtransformer_column_embedding_output_shape():
    inputs = tf.ones((16, 8, 32))
    ce = TabTransformerColumnEmbedding(num_categorical_features=8,
                                       embedding_dim=32)
    outputs = ce(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 16
    assert tf.shape(outputs)[1] == 8
    assert tf.shape(outputs)[2] == 32

