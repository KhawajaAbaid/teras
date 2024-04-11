from keras import ops
import pandas as pd
import numpy as np
from warnings import warn
from teras._src.typing import DataFrameOrNdArray
from teras._src.api_export import teras_export


@teras_export("teras.utils.compute_cardinalities")
def compute_cardinalities(x, categorical_idx: list,
                          ordinal_encoded: bool = True):
    """
    Compute cardinalities for features in the given dataset/dataframe.
    For numerical features, 0 is used as a placeholder.

    Args:
        x: Input dataset or dataframe.
        categorical_idx: list, a list of indices of categorical features
            in the given dataset.
        ordinal_encoded: `bool`, Whether the categorical values have been
            ordinal encoded. Defaults to True.

    Returns:
        A 1d numpy array of cardinalities of all features.
        For numerical features, a value of 0 is used.
    """
    if isinstance(x, pd.DataFrame):
        x = x.values

    cardinalities = np.array([], dtype=np.uint16)
    for idx in range(ops.shape(x)[1]):
        if idx in categorical_idx:
            feature = ops.convert_to_numpy(x[:, idx])
            if ordinal_encoded:
                num_categories = np.max(feature) + 1
            else:
                num_categories = len(np.unique(feature))
            cardinalities = np.append(cardinalities, num_categories)
        else:
            # it's a numerical feature, in which case we append 0
            cardinalities = np.append(cardinalities, 0)
    return cardinalities


@teras_export("teras.utils.get_metadata_for_embedding")
def get_metadata_for_embedding(dataframe: pd.DataFrame,
                               categorical_features=None,
                               numerical_features=None):
    # TODO:
    #   Add support for TensorFlow datasets and PyTorch DataLoaders/Datasets
    """
    Utility function that create metadata for features in a given dataframe
    required by the Categorical and Numerical embedding layers in Teras.
    For numerical features, it maps each feature name to feature index.
    For categorical features, it maps each feature name to a tuple of
    feature index and vocabulary of words in that categorical feature.
    This metadata is usually required by the architectures that create
    embeddings of Numerical or Categorical features,
    such as TabTransformer, TabNet, FT-Transformer, etc.

    Args:
        dataframe: Input dataframe
        categorical_features: List of names of categorical features in the
            input dataset
        numerical_features: List of names of categorical features in the
            input dataset

    Returns:
        A dictionary which contains sub-dictionaries for categorical and
        numerical features where categorical dictionary is a mapping of
        categorical feature names to a tuple of feature indices and the
        lists of unique values (vocabulary) in them, while numerical
        dictionary is a mapping of numerical feature names to their indices
        {feature_name: (feature_idx, vocabulary)} for feature in categorical features.
        {feature_name: feature_idx} for feature in numerical features.
    """
    if categorical_features is None and numerical_features is None:
        raise ValueError(
            "Both `categorical_features` and `numerical_features` cannot "
            "be None at the same time. "
            "You must pass value for at least one of them. "
            f"Received, `categorical_features`: {categorical_features}, "
            f"`numerical_features`: {numerical_features}")
    categorical_features = [] if categorical_features is None else categorical_features
    numerical_features = [] if numerical_features is None else numerical_features
    features_meta_data = {}
    categorical_features_metadata = {}
    numerical_features_metadata = {}
    # Verify all specified features are present in the dataframe
    specified_columns = set(numerical_features).union(set(categorical_features))
    not_found_in_dataframe = specified_columns - set(dataframe.columns)
    if len(not_found_in_dataframe) > 0:
        raise ValueError(
            f"Following specified features not found in the dataframe, "
            f"{not_found_in_dataframe}")
    for idx, col in enumerate(dataframe.columns):
        if categorical_features is not None and col in categorical_features:
            vocabulary = sorted(list(dataframe[col].unique()))
            categorical_features_metadata.update({col: (idx, vocabulary)})
        elif numerical_features is not None and col in numerical_features:
            numerical_features_metadata.update({col: idx})

    features_meta_data["categorical"] = categorical_features_metadata
    features_meta_data["numerical"] = numerical_features_metadata
    return features_meta_data


@teras_export("teras.utils.convert_tf_dict_to_array_tensor")
def convert_tf_dict_to_array_tensor(dict_tensor):
    """
    Converts a batch of data taken from tensorflow dictionary format
    dataset to array format.
    Args:
        dict_tensor: A batch of data taken from tensorflow dictionary
        format dataset.

    Returns:
        Array format data.
    """
    if not isinstance(dict_tensor, dict):
        warn("Given tensor is not in dictionary format."
             "Hence no processing will be applied. \n"
             f"Expected type: {dict}, Received type: {type(dict_tensor)}")
        return

    feature_names = dict_tensor.keys()
    array_tensor = [ops.expand_dims(dict_tensor[feature_name], axis=1)
                    for feature_name in feature_names]
    array_tensor = ops.concatenate(array_tensor, axis=1)
    return array_tensor


@teras_export("teras.utils.inject_missing_values")
def inject_missing_values(x: DataFrameOrNdArray,
                          miss_rate=0.1
                          ):
    """
    Injects missing (np.nan) values in the given dataframe or ndarray.

    Args:
        x: A pandas dataframe or ndarray.
        miss_rate: The fraction of missing values that should be introduced.
            Should be between 0-1. Defaults to 0.1

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

    mask = np.random.binomial(1, 1-miss_rate, size=x.shape)
    x_with_missing_data[mask == 0] = np.nan

    if is_dataframe:
        x_with_missing_data = pd.DataFrame(x_with_missing_data,
                                           columns=x.columns)
    return x_with_missing_data


@teras_export("teras.utils.inject_missing_values")
def generate_fake_gemstone_data(num_samples: int = 16):
    """
    Generate fake gemstone like data of specified num_samples.

    Args:
        num_samples:
            Number of samples to generate

    Returns:
        A pandas DataFrame of fake gemstone like data.
    """
    fake_gem_df = pd.DataFrame({
        "cut": np.random.randint(low=0, high=3, size=(num_samples,)),
        "color": np.random.randint(low=0, high=5, size=(num_samples,)),
        "clarity": np.random.randint(low=0, high=4, size=(num_samples,)),
        "depth": np.random.randint(low=0, high=100, size=(num_samples,)),
        "table": np.random.randint(low=0, high=100, size=(num_samples,))
    })
    fake_gem_df = fake_gem_df.astype(np.float32)
    return fake_gem_df


@teras_export("teras.utils.clean_reloaded_config_data")
def clean_reloaded_config_data(x):
    """
    Cleans reloaded dictionary/list config data in the `from_config` method.

    Args:
        x: dict or list to clean.
    """
    if not isinstance(x, (dict, list)):
        return x
    if isinstance(x, dict):
        if "config" in x.keys():
            return x["config"]["value"]
        for key, value in x.items():
            x[key] = clean_reloaded_config_data(value)
        return x
    if isinstance(x, list):
        for i, value in enumerate(x):
            x[i] = clean_reloaded_config_data(value)
        return x
