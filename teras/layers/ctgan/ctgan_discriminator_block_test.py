import tensorflow as tf
from teras.layers.ctgan.ctgan_discriminator_block import CTGANDiscriminatorBlock


def test_ctgan_discriminator_block_valid_call():
    disc_block = CTGANDiscriminatorBlock(units=32)
    inputs = tf.ones((8, 5), dtype=tf.float32)
    outputs = disc_block(inputs)


def test_ctgan_discriminator_block_output_shape():
    disc_block = CTGANDiscriminatorBlock(units=32)
    inputs = tf.ones((8, 5), dtype=tf.float32)
    outputs = disc_block(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 32
