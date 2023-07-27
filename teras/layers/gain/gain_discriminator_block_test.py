import tensorflow as tf
from teras.layers.gain.gain_discriminator_block import GAINDiscriminatorBlock


def test_gain_discriminator_block_valid_call():
    gen_block = GAINDiscriminatorBlock(units=32)
    inputs = tf.ones((8, 5), dtype=tf.float32)
    outputs = gen_block(inputs)


def test_gain_discriminator_block_output_shape():
    gen_block = GAINDiscriminatorBlock(units=32)
    inputs = tf.ones((8, 5), dtype=tf.float32)
    outputs = gen_block(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 32


