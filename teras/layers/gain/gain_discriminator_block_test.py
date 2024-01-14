from keras import ops
from teras.layers.gain.gain_discriminator_block import GAINDiscriminatorBlock


def test_gain_discriminator_block_valid_call():
    gen_block = GAINDiscriminatorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = gen_block(inputs)


def test_gain_discriminator_block_output_shape():
    gen_block = GAINDiscriminatorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = gen_block(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 32


