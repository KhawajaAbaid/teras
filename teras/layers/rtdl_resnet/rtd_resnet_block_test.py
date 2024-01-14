from keras import ops
from teras.layers.rtdl_resnet.rtdl_resnet_block import RTDLResNetBlock


def test_rtdl_resnet_block_valid_call():
    resnet_block = RTDLResNetBlock(units=64)
    inputs = ops.ones((8, 4))
    outputs = resnet_block(inputs)


def test_rtdl_resnet_block_output_shape():
    resnet_block = RTDLResNetBlock(units=64)
    inputs = ops.ones((8, 4))
    outputs = resnet_block(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 4
