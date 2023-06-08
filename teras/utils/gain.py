import numpy as np
import pandas as pd


def inject_missing_values(x: pd.DataFrame,
                          miss_rate=0.1,
                          return_dataframe=True):
    """
    Injects missing (np.nan) values in the given dataset.

    Args:
        x: A pandas dataframe.
        miss_rate: The fraction of missing values that should be introduced.
            Should be between 0-1. Defaults to 0.1
        return_dataframe: If False, numpy ndarray will be returned,
            otherwise pandas dataframe will be returned.

    Returns:
        Data with missing values.

    Example:
        ```python
        data = np.arange(1000).reshape(50, 20)
        data = inject_missing_values(data, miss_rate=0.2, return_dataframe=False)
        ```
    """
    x_with_missing_data = x.copy()
    is_dataframe = isinstance(x_with_missing_data, pd.DataFrame)

    if is_dataframe:
        x_with_missing_data = x_with_missing_data.values
        is_dataframe = True

    mask = np.random.binomial(1, 1-miss_rate, size=x.shape)
    x_with_missing_data[mask == 0] = np.nan

    if return_dataframe:
        x_with_missing_data = pd.DataFrame(x_with_missing_data,
                                           columns=x.columns if is_dataframe else None)
    return x_with_missing_data
