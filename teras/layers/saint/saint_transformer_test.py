from keras import ops
from teras.layers.saint.saint_transformer import SAINTTransformer


def test_saint_transformer_output_shape():
    saint_transformer = SAINTTransformer(data_dim=10,
                                         embedding_dim=32)
    inputs = ops.ones((16, 10, 32))
    outputs = saint_transformer(inputs)
    assert len(ops.shape(outputs)) == 3
    assert ops.shape(outputs)[0] == 16
    assert ops.shape(outputs)[1] == 10
    assert ops.shape(outputs)[2] == 32
