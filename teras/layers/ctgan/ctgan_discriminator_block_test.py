from keras import ops
from teras.layers.ctgan.ctgan_discriminator_block import CTGANDiscriminatorBlock


def test_ctgan_discriminator_block_valid_call():
    disc_block = CTGANDiscriminatorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = disc_block(inputs)


def test_ctgan_discriminator_block_output_shape():
    disc_block = CTGANDiscriminatorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = disc_block(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 32
