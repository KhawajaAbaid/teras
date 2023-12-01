from keras import ops
from teras.layers.regularization import CutMix


def test_cutmix_valid_call():
    cutmix = CutMix()
    inputs = ops.ones((8, 4))
    outputs = cutmix(inputs)


def test_cutmix_output_shape():
    cutmix = CutMix()
    inputs = ops.ones((8, 4))
    outputs = cutmix(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 4
