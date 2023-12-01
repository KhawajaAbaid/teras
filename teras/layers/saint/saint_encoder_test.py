from keras import ops
from teras.layers.saint.saint_encoder import SAINTEncoder


def test_saint_encoder_output_shape():
    encoder = SAINTEncoder(data_dim=10,
                           embedding_dim=32)
    inputs = ops.ones((16, 10, 32))
    outputs = encoder(inputs)
    assert len(ops.shape(outputs)) == 3
    assert ops.shape(outputs)[0] == 16
    assert ops.shape(outputs)[1] == 10
    assert ops.shape(outputs)[2] == 32
