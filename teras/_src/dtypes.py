import numpy as np


_DEFAULT_NUMPY_FLOATX = np.float32


def get_default_numpy_floatx():
    return _DEFAULT_NUMPY_FLOATX


def set_default_numpy_floatx(dtype):
    global _DEFAULT_NUMPY_FLOATX
    _DEFAULT_NUMPY_FLOATX = dtype
