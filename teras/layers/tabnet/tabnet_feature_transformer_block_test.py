from keras import ops
from teras.layers.tabnet.tabnet_feature_transformer_block import TabNetFeatureTransformerBlock


def test_tabnet_feature_transformer_block_valid_call():
    # The first TabNetFeatureTransformerBlock must have
    # use_residual_normalization = False
    feature_transformer = TabNetFeatureTransformerBlock(units=32,
                                                        use_residual_normalization=False)
    inputs = ops.ones((8, 10))
    outputs = feature_transformer(inputs)


def test_tabnet_feature_transformer_block_output_shape():
    feature_transformer = TabNetFeatureTransformerBlock(units=32,
                                                        use_residual_normalization=False)
    inputs = ops.ones((8, 10))
    outputs = feature_transformer(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 32
