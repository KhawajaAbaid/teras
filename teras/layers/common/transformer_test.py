from teras.layers.common.transformer import FeedForward, Transformer, Encoder
from teras.layers import SAINTNumericalFeatureEmbedding
from keras import ops


def test_feedforward_output_shape():
    feed_forward = FeedForward(embedding_dim=32)
    inputs = ops.ones((128, 16), dtype="float32")
    outputs = feed_forward(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 128
    assert ops.shape(outputs)[1] == 32


def test_transformer_output_shape():
    transformer = Transformer(embedding_dim=32)
    inputs = ops.ones((128, 16, 32), dtype="float32")
    outputs = transformer(inputs)
    assert len(ops.shape(outputs)) == 3
    assert ops.shape(outputs)[0] == 128
    assert ops.shape(outputs)[1] == 16
    assert ops.shape(outputs)[2] == 32


def test_encoder_output_shape():
    encoder = Encoder(embedding_dim=32)
    inputs = ops.ones((128, 16, 32), dtype="float32")
    outputs = encoder(inputs)
    assert len(ops.shape(outputs)) == 3
    assert ops.shape(outputs)[0] == 128
    assert ops.shape(outputs)[1] == 16
    assert ops.shape(outputs)[2] == 32
