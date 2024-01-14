from keras import ops
from teras.layers.saint.saint_projection_head import SAINTProjectionHead


def test_saint_projection_head_output_shape():
    saint_transformer = SAINTProjectionHead(hidden_dim=16,
                                            output_dim=8)
    inputs = ops.ones((16, 10))
    outputs = saint_transformer(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 16
    assert ops.shape(outputs)[1] == 8
