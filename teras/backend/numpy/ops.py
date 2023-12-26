import numpy as np


def norm(x, ord, axis, keepdims):
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
