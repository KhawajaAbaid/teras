import tensorflow as tf
from teras.layers.gain import GeneratorBlock, DiscriminatorBlock


def test_generator_block_output_shape():
    gen_block = GeneratorBlock(units=32)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = gen_block(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 32


def test_discriminator_block_output_shape():
    gen_block = DiscriminatorBlock(units=32)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = gen_block(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 32
