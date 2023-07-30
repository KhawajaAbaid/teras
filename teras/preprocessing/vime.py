import tensorflow as tf
import pandas as pd
import numpy as np
from teras.preprocessing.base.base_data_transformer import BaseDataTransformer as _BaseDataTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from teras.utils.types import FeaturesNamesType
from teras.utils.vime import vime_mask_generator, vime_pretext_generator


class VimeDataTransformer(_BaseDataTransformer):
    """
    DataTransformer class for VIME archictecure.
    It one hot encodes the categorical features
    and normalizes the numerical features.

    Args:
        categorical_features: ``List[str]``,
            List of categorical features names.

        numerical_features: ``List[str]``,
            List of numerical features names.
    """
    def __init__(self,
                 categorical_features: FeaturesNamesType = None,
                 numerical_features: FeaturesNamesType = None
                 ):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.one_hot_encoder = OneHotEncoder()

    def fit(self, x: pd.DataFrame):
        if self.categorical_features is not None:
            self.one_hot_encoder.fit(x[self.categorical_features])

    def transform(self, x: pd.DataFrame):
        if self.categorical_features is not None:
            x[self.categorical_features] = self.one_hot_encoder(x[self.categorical_features])

        if self.numerical_features is not None:
            x[self.numerical_features] = np.sum(x[self.numerical_features], axis=1)
        return x


class VimeDataSampler:
    """
    VimeDataSampler class for ``VIME`` architecture.
    It prepares dataset in a format that is acceptable
    for the VIME architecure.

    Args:
        batch_size: ``int``, default 1024,
            Batch size to use.

        shuffle: ``bool``, default True,
            Whether to shuffle the dataset.

        labeled_split: ``float``, default 0.5,
            The size of the original dataset that should be used as labeled
            dataset. ``1 - labeled_split`` will be used as unlabeled dataset for
            in the `VIME` architecture.
            If you have separate unlabeled dataset, pass it along with labeled dataset
            to the  ``get_dataset`` method, in which case this parameter will be ignored
            and your labeled dataset won't be split.

    """
    def __init__(self,
                 batch_size: int = 1024,
                 shuffle: bool = True,
                 labeled_split: float = 0.5,
                 p_m: float = 0.3
                 ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.labeled_split = labeled_split
        self.p_m = p_m

    def get_dataset(self,
                    x,
                    y=None,
                    unlabeled_data=None):
        """
        It will return two datasets, one labeled, for semi-supervised training
        and the other unlabeled for self-supervised training.

        Args:
            x:
                Input dataset. It must not contain the target variable.

            y:
                Targets array for ``x`` (the labeled dataset).

            unlabeled_data:
                Unlabeled dataset. If specified, ``x`` won't be split,
                otherwise ``labeled_split`` argument specified during the
                instantiation of the ``VimeDataSampler`` class will be used
                to split ``x``.

        Returns:
            A dataset for self-supervised learning and a dataset for semi-supervised learning.
        """
        labeled_data = x
        if unlabeled_data is None:
            labeled_data, unlabeled_data = train_test_split(x,
                                                            test_size=self.labeled_split,
                                                            shuffle=True)

        # Self supervised dataset
        mask_unlabeled = vime_mask_generator(x=unlabeled_data,
                                             p_m=self.p_m
                                             )
        mask_corrupted, x_tilde = vime_pretext_generator(x=unlabeled_data,
                                                         mask=mask_unlabeled)

        self_ds = tf.data.Dataset.from_tensor_slices((x_tilde,
                                                      {"mask_estimator": mask_corrupted,
                                                       "feature_estimator": mask_unlabeled}))

        if self.batch_size is not None:
            self_ds = self_ds.batch(batch_size=self.batch_size)

        # Semi supervised dataset
        num_samples_labeled = len(labeled_data)
        num_samples_unlabeled = len(unlabeled_data)

        semi_labeled_ds = (tf.data.Dataset
                           .from_tensor_slices({"x_labeled": labeled_data, "y_labeled": y},
                                               name="labeled_dataset")
                           .shuffle(buffer_size=num_samples_labeled,
                                    reshuffle_each_iteration=True)
                           )

        semi_unlabeled_ds = (tf.data.Dataset
                             .from_tensor_slices({"x_unlabeled": unlabeled_data},
                                                 "unlabeled_dataset")
                             .shuffle(buffer_size=num_samples_unlabeled,
                                      reshuffle_each_iteration=True)
                             )

        # If there are fewer unlabeled samples than labeled samples, we'll repeat unlabeled dataset
        # because it is essential to have a batch of unlabeled dataset along with labeled dataset
        # for semi supervised training in VIME.
        # In case when unlabeled dataset contains more samples than labeled, we won't alter anything
        # Because by default, tensorflow dataset will produce batches equal to
        # `math.ceil(min(len(dataset_1), len(dataset_2)) / batch_size)`
        if num_samples_labeled > num_samples_unlabeled:
            num_times_to_repeat = tf.cast(tf.math.ceil(num_samples_labeled / num_samples_unlabeled),
                                          dtype=tf.int64)
            semi_unlabeled_ds = semi_unlabeled_ds.repeat(num_times_to_repeat)

        semi_ds = tf.data.Dataset.zip((semi_labeled_ds, semi_unlabeled_ds),
                                      "semi_supervised_dataset")
        if self.batch_size is not None:
            semi_ds = semi_ds.batch(batch_size=self.batch_size)
        return self_ds, semi_ds
