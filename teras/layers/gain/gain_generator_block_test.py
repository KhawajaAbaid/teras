import tensorflow as tf
from teras.layers.gain.gain_generator_block import GAINGeneratorBlock


def test_gain_generator_block_valid_call():
    gen_block = GAINGeneratorBlock(units=32)
    inputs = tf.ones((8, 5), dtype=tf.float32)
    outputs = gen_block(inputs)


def test_gain_generator_block_output_shape():
    gen_block = GAINGeneratorBlock(units=32)
    inputs = tf.ones((8, 5), dtype=tf.float32)
    outputs = gen_block(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 32
