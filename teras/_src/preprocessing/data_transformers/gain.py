import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from teras._src.preprocessing.data_transformers.data_transformer import DataTransformer
from teras._src.typing import FeaturesNamesType
import json


class GAINDataTransformer(DataTransformer):
    """
    GAINDataTransformer class that performs the required transformations
    on the raw dataset required by the GAIN architecture.

    Args:
        categorical_features: list, List of categorical features names in the
            dataset. Categorical features are encoded by ordinal encoder method.
            And then MinMax normalization is applied.
        continuous_features: list, List of numerical features names
            in the dataset. Numerical features are encoded using MinMax
            normalization.
    """
    def __init__(self,
                 categorical_features: FeaturesNamesType = None,
                 continuous_features: FeaturesNamesType = None
                 ):
        super().__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self._encoder = OrdinalEncoder()
        self._min_vals = None
        self._max_vals = None
        self._ordered_features_names_all = None
        self._fitted = False

    def fit(self, x: pd.DataFrame):
        # min-max normalize the entire dataset regardless of the features types.
        # For categorical, first need to encode values ordinally
        x_temp = x.copy()
        if self.categorical_features is not None:
            self._encoder.fit(x[self.categorical_features])
            x_temp[self.categorical_features] = self._encoder.transform(
                x[self.categorical_features])
        self._min_vals = np.nanmin(x_temp, axis=0)
        self._max_vals = np.nanmax(x_temp, axis=0)
        self._fitted = True

    def transform(self, x: pd.DataFrame):
        """
        Transforms the data (applying normalization etc)
        and returns a tensorflow dataset.
        It also stores the metadata of features
        that is used in the reverse transformation step.

        Args:
            x: Data to transform. Must be a pandas DataFrame.

        Returns:
            Transformed data.
        """
        if not self._fitted:
            raise AssertionError(
                "You haven't yet fitted the DataTransformer. "
                "You must call the `fit` method before you can call the "
                "`transform` method. ")
        if not isinstance(x, pd.DataFrame):
            raise ValueError(
                "Only pandas dataframe is supported by DataTransformer class."
                f" But data of type {type(x)} was passed. "
                f"Please convert it to pandas dataframe before passing.")
        self._ordered_features_names_all = x.columns

        if self.categorical_features is not None:
            x[self.categorical_features] = self._encoder.transform(
                x[self.categorical_features])

        x = (x - self._min_vals) / self._max_vals
        return x

    def inverse_transform(self, x):
        """
        Inverse Transforms the transformed data.

        Args:
            x: Transformed Data.

        Returns:
            Pandas dataframe of data in its original scale
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=self._ordered_features_names_all)

        if not self._fitted:
            raise AssertionError(
                "You haven't yet fitted the DataTransformer. "
                "You must call the `fit` method before you can call the "
                "`inverse_transform` method. ")

        # min-max transformation was applied to the whole dataset even to the
        # ordinal encoded categorical features, so first, undo that
        # transformation and then inverse transform categorical
        x = (x * self._max_vals) + self._min_vals
        if self.categorical_features is not None:
            x[self.categorical_features] = self._encoder.inverse_transform(
                x[self.categorical_features])
        return x

    def save(self, filename):
        """
        Saves the fitted state of `DataTransformer` instance for portability,
        in the `json` format.

        Args:
            filename: Filename or file path ending in `.json` extension.
        """
        args = {
            "categorical_features": self.categorical_features,
            "continuous_features": self.continuous_features
        }
        attrs = {
            "_min_vals": list(self._min_vals),
            "_max_vals": list(self._max_vals),
            "_ordered_features_names_all": self._ordered_features_names_all,
            "_fitted": self._fitted
        }
        encoder_attrs = {
            "categories_": list(self._encoder.categories_)
        }
        state = {
            "args": args,
            "attrs": attrs,
            "encoder_attrs": encoder_attrs,
        }
        with open(filename, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, filename):
        """
        Loads the saved state of `DataTransformer` from the `json` file.

        Args:
            filename: Filename or file path ending in `.json` extension.

        Returns:
            An instance of `GAINDataTransformer` with state stored in the
            `filename` json file.
        """
        with open(filename, "r") as f:
            state = json.load(f)
        c = cls(**state.pop("params"))
        for name, value in state.pop("attrs"):
            c.__setattr__(name, value)
        for name, value in state.pop("encoder_attrs"):
            c._encoder.__setattr__(name, value)
        return c
