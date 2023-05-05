from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Union


LayerType = Union[str, layers.Layer]

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