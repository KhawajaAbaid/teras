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