# TabTransformer Utility Function(s)
import pandas as pd
import tensorflow as tf


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
        categorical_features_vocab[cat_feat] = tf.constant(sorted(list(inputs[cat_feat].unique())))
    return categorical_features_vocab


def dataframe_to_tf_dataset(
                    dataframe: pd.DataFrame,
                    target: str = None,
                    shuffle: bool = True,
                    batch_size: int = 1024,
                    ):
    """
    Builds a tf.data.Dataset from a given pandas dataframe

    Args:
        dataframe: A pandas dataframe
        target: Name of the target column
        shuffle: Whether to shuffle the dataset
        batch_size: Batch size

    Returns:
         A tf.data.Dataset dataset
    """
    df = dataframe.copy()
    if target:
        labels = df.pop(target)
        dataset = {}
        for key, value in df.items():
            dataset[key] = value[:, tf.newaxis]
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    else:
        dataset = {}
        for key, value in df.items():
            dataset[key] = value[:, tf.newaxis]
        dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset