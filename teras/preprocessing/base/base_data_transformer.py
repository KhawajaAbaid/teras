from abc import abstractmethod
import pandas as pd


class BaseDataTransformer:
    """
    Base class for ``DataTransformer`` classes.
    It provides the common methods and attributes.
    """
    def __init__(self):
        self.metadata = dict()
        self.metadata["numerical"] = dict()
        self.metadata["categorical"] = dict()
        self.fitted = False

    @abstractmethod
    def fit(self, x, **kwargs):
        pass

    @abstractmethod
    def transform(self, x, **kwargs):
        pass

    def get_metadata(self):
        """
        Returns:
            named tuple of features metadata.
        """
        return self.metadata

    def fit_transform(self,
                      x: pd.DataFrame,
                      return_dataframe: bool = True):
        self.fit(x)
        return self.transform(x, return_dataframe=return_dataframe)
