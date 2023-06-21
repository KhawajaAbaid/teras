from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Union, Tuple
import pandas as pd


LayerType = Union[str, layers.Layer]
FEATURE_NAMES_TYPE = Union[List[str], Tuple[str]]


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



def get_activation(activation: LayerType,
                   units=None):
    """
    Retrieves and returns a keras activation function if not already.

    Args:
        activation: default None.
            If type of activation is a keras function, it is returned as is.
            If it is of type, str, that is, it is a name,
            then relevant activation function is returned

    Returns:
        Keras Activation function
    """
    if isinstance(activation, str):
        activation = activation.lower()
        try:
            activation_func = keras.activations.get(activation)
        except ValueError:
            raise ValueError(f"{activation} function's name is either wrong or this activation is not supported by Keras."
                             f"Please contact Teras team, and we'll make sure to add this to Teras.")
    else:
        activation_func = activation
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
        target: str = None,
        shuffle: bool = True,
        batch_size: int = 1024,
        as_dict: bool = False,
):
    """
    Builds a tf.data.Dataset from a given pandas dataframe

    Args:
        dataframe: A pandas dataframe
        target: Name of the target column
        shuffle: Whether to shuffle the dataset
        batch_size: Batch size
        as_dict: Whether to make a tensorflow dataset in a dictionary format
            where each record is a mapping of features names against their values.

            SOME GUIDELINES on when to create dataset in dictionary format and when not:
            1. If your dataset is composed of heterogeneous data formats, i.e. it contains
                features where some features contain integers/floats AND others contain strings,
                and you don't want to manually encode the string values into integers/floats,
                then your dataset must be in dictionary format, which you can get by setting
                the `as_dict` parameter to `True`.


    Returns:
         A tf.data.Dataset dataset
    """
    df = dataframe.copy()
    if target:
        labels = df.pop(target)
        if as_dict:
            dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        else:
            df = df.values
            labels = labels.values
            dataset = tf.data.Dataset.from_tensor_slices((df, labels))
    else:
        if as_dict:
            dataset = tf.data.Dataset.from_tensor_slices(dict(df))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(df.values)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset
