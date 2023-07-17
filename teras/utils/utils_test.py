from teras.utils.utils import dataframe_to_tf_dataset
import pandas as pd
import numpy as np


def test_dataframe_to_tf_dataset_without_label():
    df = pd.DataFrame(data={"length": np.ones(10),
                            "area": np.ones(10) * 2})
    ds = dataframe_to_tf_dataset(df)


def test_dataframe_to_tf_dataset_with_single_label():
    df = pd.DataFrame(data={"length": np.ones(10),
                            "area": np.ones(10) * 2})
    ds = dataframe_to_tf_dataset(df, target="area")


def test_dataframe_to_tf_dataset_with_multiple_labels():
    df = pd.DataFrame(data={"length": np.ones(10),
                            "area": np.ones(10) * 2,
                            "perimeter": np.ones(10) * 3})
    ds = dataframe_to_tf_dataset(df, target=["area", "perimeter"])
