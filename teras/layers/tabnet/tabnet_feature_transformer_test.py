from keras import ops
from teras.layers.tabnet.tabnet_feature_transformer import TabNetFeatureTransformer


def test_tabnet_feature_transformer_valid_call():
    TabNetFeatureTransformer.reset_shared_layers()
    feature_transformer = TabNetFeatureTransformer(units=32)
    inputs = ops.ones((8, 10))
    outputs = feature_transformer(inputs)


def test_tabnet_feature_transformer_output_shape():
    TabNetFeatureTransformer.reset_shared_layers()
    feature_transformer = TabNetFeatureTransformer(units=32)
    inputs = ops.ones((8, 10))
    outputs = feature_transformer(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 32
