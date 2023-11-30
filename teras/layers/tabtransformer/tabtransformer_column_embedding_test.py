from keras import ops
from teras.layers.tabtransformer.tabtransformer_column_embedding import TabTransformerColumnEmbedding


def test_tabtransformer_column_embedding_output_shape():
    inputs = ops.ones((16, 8, 32))
    ce = TabTransformerColumnEmbedding(num_categorical_features=8,
                                       embedding_dim=32)
    outputs = ce(inputs)
    assert len(ops.shape(outputs)) == 3
    assert ops.shape(outputs)[0] == 16
    assert ops.shape(outputs)[1] == 8
    assert ops.shape(outputs)[2] == 32

