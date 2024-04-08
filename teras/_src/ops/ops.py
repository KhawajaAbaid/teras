import keras

from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.ops.norm")
def norm(x: keras.KerasTensor,
         ord: str = None,
         axis: int = None,
         keepdims: bool = False
         ):
    """
    Matrix or vector norm.

    Args:
        x: Input tensor.
        ord: Order of norm.
        axis: If axis is an integer, it specifies the axis of x along
            which to compute the vector norms. If axis is a 2-tuple,
            it  specifies the axes that hold 2-D matrices, and the matrix
            norms of these matrices are computed. If axis is None then
            either a vector norm (when x is 1-D) or a matrix norm
            (when x is 2-D) is returned. The default is None.
        keepdims: If this is set to True, the axes which are normed over
            are left in the result as dimensions with size one. With this
            option the result will broadcast correctly against the
            original x.

    Returns:
        Norm of the matrix or vector(s).
    """
    return backend.norm(x, ord=ord, axis=axis, keepdims=keepdims)
