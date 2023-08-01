from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import List, Union, Tuple
import pandas as pd
import numpy as np
from warnings import warn
from teras.utils.types import ActivationType
from teras import activations

LayerType = Union[str, layers.Layer]
LAYERS_COLLECTION = Union[List[layers.Layer], layers.Layer, models.Model]
LAYERS_CONFIGS_COLLECTION = Union[List[dict], dict]


def tf_random_choice(inputs,
                     n_samples: int,
                     p: List[float] = None):
    """Tensorflow equivalent of np.random.choice

    Args:
        inputs: Tensor to sample from
        n_samples: Number of samples to draw
        p: Probabilities for each sample

    Reference(s):
        https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
    """
    probs = p
    if probs is None:
        probs = tf.random.uniform([n_samples])
    # Normalize probabilities so they all sum to one
    probs /= tf.reduce_sum(probs)
    probs = tf.expand_dims(probs, 0)
    # Convert them to log probabilities because that's what the tf.random.categorical function expects
    probs = tf.math.log(probs)
    indices = tf.random.categorical(probs, n_samples, dtype=tf.int32)
    return tf.gather(inputs, indices)


def get_normalization_layer(normalization: LayerType) -> layers.Layer:
    """
    Creates and returns a keras normalization layer if not already.

    Args:
        normalization: default BatchNormalization.
            If type of normalization is a keras layer, it is returned as is.
            If it is of type, str, that is, it is a name,
            then relevant normalization layer is returned

    Returns:
        Keras Normalization Layer
    """
    if isinstance(normalization, str):
        normalization = normalization.lower()
        if normalization in ["batchnormalization", "batch"]:
            normalization_layer = layers.BatchNormalization()
        elif normalization in ["layernormalization", "layer"]:
            normalization_layer = layers.LayerNormalization()
        elif normalization in ["unitnormalization", "unit"]:
            normalization_layer = layers.UnitNormalization()
        elif normalization in ["groupnormalization", "group"]:
            normalization_layer = layers.GroupNormalization()
        else:
            raise ValueError(f"Normalization type {normalization} is not supported in Keras.")
    elif isinstance(normalization, layers.Layer):
        normalization_layer = normalization
    else:
        raise ValueError(f"Invalid Normalization value type. Expected type str or keras.layers.Layer"
                         f" Received value {normalization} of type {type(normalization)}")
    return normalization_layer


def get_activation(activation: ActivationType):
    """
    Retrieves and returns a keras activation function if not already.

    Args:
        activation:
            If type of activation is a keras function, it is returned as is.
            If it is of type, str, that is, it is a name,
            then relevant activation function is returned, if it is offered
            by either ``Teras`` or ``Keras``, otherwise an error is raised.

    Returns:
        Keras Activation function
    """
    if isinstance(activation, str):
        activation = activation.lower()
        # First check if Keras offers that function
        try:
            activation_func = keras.activations.get(activation)
        except ValueError:
            # Then check if Teras offers it
            if activation == "glu":
                activation_func = activations.glu
            elif activation == "geglu":
                activation_func = activations.geglu
            elif activation == "gumblesoftmax":
                activation_func = activations.gumbel_softmax
            elif activation == "sparsemax":
                activation_func = activations.sparsemax
            elif activation == "sparsemoid":
                from teras.utils.node import sparsemoid
                activation_func = sparsemoid
            else:
                # Otherwise return error
                raise ValueError(f"{activation} function's name is either incorrect "
                                 f"or this activation is not offered by either Keras and Teras. ")
    elif callable(activation):
        activation_func = activation

    else:
        TypeError("Unsupported type for `activation` argument. "
                  f"Expected type(s): [`str`, `callable`]. Received: {type(activation)}")
    return activation_func


def get_initializer(initializer: LayerType):
    """
    Retrieves and returns a keras initializer function if not already.

    Args:
        initializer: default uniform.
            If type of initializer is a keras function, it is returned as is.
            If it is of type, str, that is, it is a name,
            then relevant initializer function is returned

    Returns:
        Keras initializer function
    """
    if isinstance(initializer, str):
        initializer = initializer.lower()
        try:
            initializer_func = keras.initializers.get(initializer)
        except ValueError:
            raise ValueError(f"{initializer} function's name is either wrong or this activation is not supported by Keras."
                             f"Please contact Teras team, and we'll make sure to add this to Teras.")
    else:
        initializer_func = initializer
    return initializer_func


def get_categorical_features_cardinalities(dataframe,
                                           categorical_features):
    cardinalities = []
    for feature in categorical_features:
        cardinalities.append(dataframe[feature].nunique())
    return cardinalities


def get_features_metadata_for_embedding(dataframe: pd.DataFrame,
                                        categorical_features=None,
                                        numerical_features=None):
    """
    Utility function that create metadata for features in a given dataframe
    required by the Categorical and Numerical embedding layers in Teras.
    For numerical features, it maps each feature name to feature index.
    For categorical features, it maps each feature name to a tuple of
    feature index and vocabulary of words in that categorical feature.
    This metadata is usually required by the architectures that create embeddings
    of Numerical or Categorical features,
    such as TabTransformer, TabNet, FT-Transformer, etc.

    Args:
        dataframe: Input dataframe
        categorical_features: List of names of categorical features in the input dataset
        numerical_features: List of names of categorical features in the input dataset

    Returns:
        A dictionary which contains sub-dictionaries for categorical and numerical features
        where categorical dictionary is a mapping of categorical feature names to a tuple of
        feature indices and the lists of unique values (vocabulary) in them,
        while numerical dictionary is a mapping of numerical feature names to their indices.
        {feature_name: (feature_idx, vocabulary)} for feature in categorical features.
        {feature_name: feature_idx} for feature in numerical features.
    """
    if categorical_features is None and numerical_features is None:
        raise ValueError("Both `categorical_features` and `numerical_features` cannot be None at the same time. "
                         "You must pass value for at least one of them. If your dataset contains both types of "
                         "features then it is strongly recommended to pass features names for both types. "
                         f"Received, `categorical_features`: {categorical_features}, "
                         f"`numerical_features`: {numerical_features}")
    features_meta_data = {}
    categorical_features_metadata = {}
    numerical_features_metadata = {}
    for idx, col in enumerate(dataframe.columns):
        if categorical_features is not None:
            if col in categorical_features:
                vocabulary = sorted(list(dataframe[col].unique()))
                categorical_features_metadata.update({col: (idx, vocabulary)})
        if numerical_features is not None:
            if col in numerical_features:
                numerical_features_metadata.update({col: idx})

    features_meta_data["categorical"] = categorical_features_metadata
    features_meta_data["numerical"] = numerical_features_metadata
    return features_meta_data


def dataframe_to_tf_dataset(
        dataframe: pd.DataFrame,
        target: Union[str, list] = None,
        shuffle: bool = True,
        batch_size: int = 1024,
):
    """
    Builds a tf.data.Dataset from a given pandas dataframe

    Args:
        dataframe: `pd.DataFrame`,
            A pandas dataframe
        target: `str` or `list`,
            Name of the target column or list of names of the target columns.
        shuffle: `bool`, default True
            Whether to shuffle the dataset
        batch_size: `int`, default 1024,
            Batch size

    Returns:
         A tf.data.Dataset dataset
    """
    df = dataframe.copy()
    if target is not None:
        if isinstance(target, (list, tuple, set)):
            labels = []
            for feat in target:
                labels.append(df.pop(feat).values)
            labels = tf.transpose(tf.constant(labels))
        else:
            labels = df.pop(target)
            labels = labels.values
        dataset = tf.data.Dataset.from_tensor_slices((df.values, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(df.values)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


def convert_dict_to_array_tensor(dict_tensor):
    """
    Converts a batch of data taken from tensorflow dictionary format dataset
    to array format.
    Args:
        dict_tensor: A batch of data taken from tensorflow dictionary format dataset.

    Returns:
        Array format data.
    """
    if not isinstance(dict_tensor, dict):
        warn("Given tensor is not in dictionary format. Hence no processing will be applied. \n"
             f"Expected type: {dict}, Received type: {type(dict_tensor)}")
        return

    feature_names = dict_tensor.keys()
    array_tensor = [tf.expand_dims(dict_tensor[feature_name], axis=1)
                    for feature_name in feature_names]
    array_tensor = tf.concat(array_tensor, axis=1)
    return array_tensor


def serialize_layers_collection(layers_collection: LAYERS_COLLECTION):
    """
    Serializes a collection of keras layers/models.

    Returns:
        If layers_collection is an instance of list or tuple, it returns a list of serialized layers
        otherwise it simply returns the serialized layer or model.
    """
    if isinstance(layers_collection, (list, tuple)):
        layers_collection_serialized = [keras.layers.serialize(layer)
                                        for layer in layers_collection]
    else:
        # We assume it's either of type Layer or Model, or None
        layers_collection_serialized = keras.layers.serialize(layers_collection)

    return layers_collection_serialized


def deserialize_layers_collection(layers_configs_collection: LAYERS_CONFIGS_COLLECTION):
    """
    De-serializes a collection of keras layers/models configs.

    Returns:
        If layers_configs_collection is a list of layers config dictionaries, it returns a
        list of de-serialized layers otherwise if it's a config dictionary, it returns the
         de-serialized layer or model.
    """
    if isinstance(layers_configs_collection, list):
        layers_collection_deserialized = [keras.layers.deserialize(layer)
                                          for layer in layers_configs_collection]
    else:
        # We assume it's either a dict or None
        layers_collection_deserialized = keras.layers.deserialize(layers_configs_collection)

    return layers_collection_deserialized


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
