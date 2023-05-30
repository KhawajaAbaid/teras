# NODE Utility Function(s)
import tensorflow as tf


def sparsemoid(x):
    """
    Sparsemoid function as implemented by the authors of
    Neural Oblivious Decision Tree Ensembles (NODE) paper.
    It is used as a bin function in the NODE architecture.

    Reference:
        https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
    """
    return tf.clip_by_value(0.5 * x + 0.5, 0., 1.)