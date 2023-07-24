import tensorflow as tf
from typing import Union, List, Callable
import numpy as np


# Taken from TensorFlow Addons

Number = Union[
                float,
                int,
                np.float16,
                np.float32,
                np.float64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ]


TensorLike = Union[
                List[Union[Number, list]],
                tuple,
                Number,
                np.ndarray,
                tf.Tensor,
                tf.SparseTensor,
                tf.Variable,
            ]

ActivationType = Union[str,
                       Callable,
                ]

