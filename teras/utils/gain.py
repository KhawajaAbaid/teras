import numpy as np
import pandas as pd


def introduce_missing_data_in_this_thing(x: pd.DataFrame,
                                         miss_rate=0.1,
                                         return_dataframe=True):
    """
    Args:
        x: A pandas dataframe.
        miss_rate: The fraction of missing values that should be introduced.
            Should be between 0-1. Defaults to 0.1
        return_dataframe: If False, numpy ndarray will be returned,
            otherwise pandas dataframe will be returned.
    """
    mask = np.random.binomial(1, 1-miss_rate, size=x.shape)
    x_with_missing_data = x.copy()
    if isinstance(x_with_missing_data, pd.DataFrame):
        x_with_missing_data = x_with_missing_data.values
    x_with_missing_data[mask == 0] = np.nan

    if return_dataframe:
        x_with_missing_data = pd.DataFrame(x_with_missing_data, columns=x.columns)
    return x_with_missing_data
