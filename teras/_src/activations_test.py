from keras import ops
from teras._src.activations import (glu,
                               geglu,
                               gumbel_softmax,
                               sparsemax)


# ================== glu tests ====================
def test_glu_valid_call():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = glu(inputs)


def test_glu_output_shape():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = glu(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 2    # glu halves the dimensions


# ================== geglu tests ====================
def test_geglu_valid_call():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = geglu(inputs)


def test_geglu_output_shape():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = geglu(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 2    # geglu halves the dimensions


# ================== gumbel_softmax tests ====================
def test_gumbel_softmax_valid_call():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = gumbel_softmax(inputs)


def test_gumbel_softmax_output_shape():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = gumbel_softmax(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 4


# ================== sparsemax tests ====================
def test_sparsemax_valid_call():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = gumbel_softmax(inputs)


def test_sparsemax_output_shape():
    inputs = ops.ones((8, 4), dtype="float32")
    outputs = gumbel_softmax(inputs)
    assert len(ops.shape(outputs)) == 2
    assert ops.shape(outputs)[0] == 8
    assert ops.shape(outputs)[1] == 4
