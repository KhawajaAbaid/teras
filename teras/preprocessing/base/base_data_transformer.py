from collections import namedtuple
from copy import deepcopy
from abc import abstractmethod
import pandas as pd
import numpy as np


class BaseDataTransformer:
    """
    Base class for DataTransformer with common
    methods and attributes.
    """
    def __init__(self):
        self.meta_data = dict()
        self.meta_data["numerical"] = dict()
        self.meta_data["categorical"] = dict()
        self.fitted = False

    @abstractmethod
    def fit(self, x, **kwargs):
        pass

    @abstractmethod
    def transform(self, x, **kwargs):
        pass

    def get_meta_data(self):
        """
        Returns:
            named tuple of features meta data.
        """
        MetaData = namedtuple("MetaData", self.meta_data.keys())

        CategoricalMetaData = namedtuple("CategoricalMetaData",
                                         self.meta_data["categorical"].keys())
        categorical_meta_data = CategoricalMetaData(**self.meta_data["categorical"])

        NumericalMetaData = namedtuple("NumericalMetaData",
                                       self.meta_data["numerical"].keys())
        numerical_meta_data = NumericalMetaData(**self.meta_data["numerical"])

        meta_data_copy = deepcopy(self.meta_data)
        meta_data_copy["categorical"] = categorical_meta_data
        meta_data_copy["numerical"] = numerical_meta_data
        meta_data_tuple = MetaData(**meta_data_copy)
        return meta_data_tuple

    def fit_transform(self,
                      x: pd.DataFrame,
                      return_dataframe: bool = True):
        self.fit(x)
        return self.transform(x, return_dataframe=return_dataframe)
