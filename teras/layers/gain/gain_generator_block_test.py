from keras import ops
from teras.layers.gain.gain_generator_block import GAINGeneratorBlock


def test_gain_generator_block_valid_call():
    gen_block = GAINGeneratorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = gen_block(inputs)


def test_gain_generator_block_output_shape():
    gen_block = GAINGeneratorBlock(units=32)
    inputs = ops.ones((8, 5))
    outputs = gen_block(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 32
