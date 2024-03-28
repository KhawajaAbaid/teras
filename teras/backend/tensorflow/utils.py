import tensorflow as tf
import pandas as pd
from typing import Union


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


def create_gain_dataset(x_generator, x_discriminator, seed: int = 1337):
    """
    Creates a tensorflow dataset compatible with the `GAIN` architecture.
    It also shuffles both datasets before creating the tensorflow dataset.

    Args:
        x_generator: Dataset to use for training the `GAINGenerator` model.
        x_discriminator: Dataset to use for training the `GAINDiscriminator`
        model.
        Note that, in most cases, both of these datasets will be the same.
        seed: int, seed to use in shuffling. Defaults to 1337.
    """
    return tf.data.Dataset.from_tensor_slices(
        (tf.random.shuffle(x_generator, seed=seed),
         tf.random.shuffle(x_discriminator, seed=seed + 99))
         )
