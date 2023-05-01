import tensorflow as tf


# NODE Utility Function(s)
def sparsemoid(x):
    """
    Sparsemoid function as implemented by the authors of
    Neural Oblivious Decision Tree Ensembles (NODE) paper.
    It is used as a bin function in the NODE architecture.

    Reference:
        https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
    """
    return tf.clip_by_value(0.5 * x + 0.5, 0., 1.)


# TabTransformer Utility Function(s)
def get_categorical_features_vocab(inputs,
                                   categorical_features):
    """
    Utility function for TabTransformer that creates vocabulary for the categorical feature values
    which is required by the Column Embedding layer in the TabTransformer.
    It is a preprocessing function and is called by the user.
    Args:
        inputs: Input dataset
        categorical_features: List of names of categorical features in the input dataset
    """
    categorical_features_vocab = {}
    for cat_feat in categorical_features:
        categorical_features_vocab[cat_feat] = sorted(list(inputs[cat_feat].unique()))
    return categorical_features_vocab