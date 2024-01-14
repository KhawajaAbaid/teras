from keras import ops
from teras.layers.ft_transformer.ft_cls_token import FTCLSToken


def test_ft_cls_token_valid_call():
    inputs = ops.ones((8, 10, 16))
    cls_token = FTCLSToken(embedding_dim=16)
    outputs = cls_token(inputs)


def test_ft_cls_token_output_shape():
    inputs = (ops.ones((8, 10, 16)))
    cls_token = FTCLSToken(embedding_dim=16)
    outputs = cls_token(inputs)

    assert len(ops.shape(outputs)) == 3
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 11
    assert ops.shape(outputs)[2] == 16
