from keras import ops
from teras.layers.ctgan.ctgan_generator_block import CTGANGeneratorBlock


def test_ctgan_generator_block_valid_call():
    gen_block = CTGANGeneratorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = gen_block(inputs)


def test_ctgan_generator_block_output_shape():
    gen_block = CTGANGeneratorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = gen_block(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 32 + 5
