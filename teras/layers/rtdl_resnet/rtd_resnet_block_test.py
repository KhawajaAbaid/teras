import tensorflow as tf
from teras.layers.rtdl_resnet.rtdl_resnet_block import RTDLResNetBlock


def test_rtdl_resnet_block_valid_call():
    resnet_block = RTDLResNetBlock(units=64)
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = resnet_block(inputs)


def test_rtdl_resnet_block_output_shape():
    resnet_block = RTDLResNetBlock(units=64)
    inputs = tf.ones((8, 4), dtype=tf.float32)
    outputs = resnet_block(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 4
