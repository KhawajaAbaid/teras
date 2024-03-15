from abc import abstractmethod
import pandas as pd


class DataTransformer:
    """
    Base `DataTransformer` class.
    It provides the common methods and attributes.
    """
    def __init__(self):
        self._metadata = dict()
        self._metadata["continuous"] = dict()
        self._metadata["categorical"] = dict()
        self._fitted = False

    @abstractmethod
    def fit(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def transform(self, x):
        raise NotImplementedError

    def get_metadata(self):
        """
        Returns:
            named tuple of features metadata.
        """
        if not self._fitted:
            raise AssertionError(
                "The `fit` method has not yet been called. Please fit the "
                "`DataTransformer` before accessing the `get_metadata` method."
            )
        return self._metadata

    def fit_transform(self,
                      x: pd.DataFrame):
        self.fit(x)
        return self.transform(x)

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, filename):
        raise NotImplementedError
