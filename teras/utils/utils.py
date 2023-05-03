import tensorflow as tf
from typing import List

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