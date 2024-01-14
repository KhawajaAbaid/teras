from keras import ops
from teras.layers.tabnet.tabnet_encoder import TabNetEncoder


def test_tabnet_encoder_valid_call():
    encoder = TabNetEncoder(data_dim=10,
                            feature_transformer_dim=32,
                            decision_step_output_dim=32)
    inputs = ops.ones((8, 10))
    outputs = encoder(inputs)


def test_tabnet_encoder_output_shape():
    encoder = TabNetEncoder(data_dim=10,
                            feature_transformer_dim=32,
                            decision_step_output_dim=32)
    inputs = ops.ones((8, 10))
    outputs = encoder(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 32
