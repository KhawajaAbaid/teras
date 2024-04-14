import numpy as np


np_floatx = np.float32
mask_dtype = np.uint8


def set_np_floatx(dtype):
    global np_floatx
    np_floatx = dtype


def set_mask_dtype(dtype):
    global mask_dtype
    mask_dtype = dtype
