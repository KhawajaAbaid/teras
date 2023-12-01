from keras import ops
from teras.layers.tabnet.tabnet_attentive_transformer import TabNetAttentiveTransformer


def test_tabnet_attentive_transformer_valid_call():
    attentive_transformer = TabNetAttentiveTransformer(data_dim=5)
    inputs = ops.ones((8, 5))
    prior_scales = ops.ones(ops.shape(inputs))
    outputs = attentive_transformer(inputs, prior_scales)


def test_tabnet_attentive_transformer_output_shape():
    attentive_transformer = TabNetAttentiveTransformer(data_dim=5)
    inputs = ops.ones((8, 5))
    prior_scales = ops.ones(ops.shape(inputs))
    outputs = attentive_transformer(inputs, prior_scales)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 5
