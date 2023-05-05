import tensorflow as tf
import numpy as np


# @tf.function
def mask_generator(p_m, X, batch_size=None):
    """
    Generates mask vector for self and semi-supervised learning
    Args:
        p_m: corruption probability
        X: feature matrix

    Returns:
        mask: binary mask matrix

    """
    num_samples, dim = X.shape
    mask = np.random.binomial(1, p_m, (num_samples, dim))
    return mask

def pretext_generator(m, X, batch_size=None):
    """
    Generates corrupted samples for self and semi-supervised learning

    Args:
        m: mask matrix
        X: feature matrix
        batch_size: When X is passed from within the keras model during training, its batch dimension is none, to handle
        that particular case, pass batch_size explicity

    Returns:
        m_new: final mask matrix after corruption
        x_tilde: corrupted feature matrix
    """
    m = tf.cast(m, dtype="float32")
    X = tf.cast(X, dtype="float32")
    num_samples, dim = X.shape
    if num_samples is None:
        num_samples = batch_size

    X_bar = []
    for i in range(dim):
        idx = np.random.permutation(num_samples)
        X_bar.append(tf.gather(X[:, i], idx))
    X_bar = tf.stack(X_bar, axis=1)

    # Corrupt Samples
    X_tilde = X * (1 - m) + X_bar * m

    m_new = (X != X_tilde)
    m_new = tf.cast(m_new, dtype="float32")
    return m_new, X_tilde
