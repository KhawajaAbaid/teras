from keras import ops
from teras.layers.regularization import MixUp


def test_mixup_valid_call():
    mixup = MixUp()
    inputs = ops.ones((8, 4))
    outputs = mixup(inputs)


def test_mixup_output_shape():
    mixup = MixUp()
    inputs = ops.ones((8, 4))
    outputs = mixup(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 4
