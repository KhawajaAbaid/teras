import keras
from typing import Union, List, Callable, Tuple
import numpy as np
import pandas as pd


# Taken from TensorFlow Addons - Sorry!

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


ActivationType = Union[str,
                       Callable,
                       keras.layers.Layer,
                ]


IntegerSequence = Union[List[int], Tuple[int]]

LayersList = List[keras.layers.Layer]

LayersCollection = Union[keras.layers.Layer,
                         List[keras.layers.Layer],
                         keras.Model]

NormalizationType = Union[keras.layers.Layer, str]

FloatSequence = Union[List[float],
                      Tuple[float]]

IntegerSequence = Union[List[int],
                        Tuple[int]]

InitializerType = Union[str, keras.initializers.Initializer]


LayerOrModelType = Union[keras.layers.Layer, keras.Model]

FeaturesNamesType = Union[List[str], Tuple[str]]

OptionalIntOrStr = Union[int, str, None]

ListOrArray = Union[list, np.array]

DataFrameOrNdArray = Union[pd.DataFrame, np.ndarray]
