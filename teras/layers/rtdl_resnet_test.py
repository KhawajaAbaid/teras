import tensorflow as tf
from teras.layers.rtdl_resnet import ResNetBlock


def test_rtdl_resnet_block_output_shape():
    resnet_block = ResNetBlock(units=64)
    inputs = tf.ones((128, 16), dtype=tf.float32)
    outputs = resnet_block(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16

