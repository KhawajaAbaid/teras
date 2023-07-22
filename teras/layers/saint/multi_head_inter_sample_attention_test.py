import tensorflow as tf
from teras.layers.saint.multi_head_inter_sample_attention import MultiHeadInterSampleAttention


def test_saint_multi_head_inter_sample_attention_output_shape():
    multihead_attention = MultiHeadInterSampleAttention(key_dim=32)
    inputs = tf.ones(shape=(16, 10, 32))
    outputs = multihead_attention(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 16   # number of items in each column
    assert tf.shape(outputs)[1] == 10
    assert tf.shape(outputs)[2] == 32
