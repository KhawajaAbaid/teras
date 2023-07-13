import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import OrdinalEncoder
from teras.preprocessing.base.base_data_transformer import BaseDataTransformer


FEATURE_NAMES_TYPE = List[str]


class DataTransformer(BaseDataTransformer):
    """
    DataTransformer class that performs the required transformations
    on the raw dataset required by the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        categorical_features: A list of names of categorical features.
            Categorical features are encoded by ordinal encoder method.
            And then MinMax normalization is applied.
        numerical_features: A list of names of categorical/continuous features.
            Numerical features are encoded using MinMax normalization.

    Example:
        ```python
            input_df = pd.DataFrame(np.arange(10, 3).reshape(20, 5),
                                    columns=['A', 'B', 'C'])
            numerical_features = input_df.columns
            data_transformer = DataTransformer(categorical_features=None,
                                               numerical_features=numerical_features)
            transformed_df = data_transformer.transform(input_df)
        ```
    """
    def __init__(self,
                 categorical_features: FEATURE_NAMES_TYPE = None,
                 numerical_features: FEATURE_NAMES_TYPE = None):
        super().__init__()
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.encoder = OrdinalEncoder()
        self.min_vals = None
        self.max_vals = None

    def fit(self, x,
            **kwargs):
        # We min max normalize the entire dataset regardless of the
        # features types - so we first need to have the ordinally
        # encoded values for the categorical features
        # So we create a temporary copy which contains these
        # encoded values so we can compute the min_vals and max_vals
        # for these features
        x_temp = x.copy()
        if self.categorical_features is not None:
            self.encoder.fit(x[self.categorical_features])
            x_temp[self.categorical_features] = self.encoder.transform(x[self.categorical_features])

        # if self.numerical_features is not None:
        #     self.min_vals = np.nanmin(x[self.numerical_features], axis=0)
        #     self.max_vals = np.nanmax(x[self.numerical_features], axis=0)
        self.min_vals = np.nanmin(x_temp, axis=0)
        self.max_vals = np.nanmax(x_temp, axis=0)
        self.fitted = True

    def transform(self,
                  x: pd.DataFrame,
                  **kwargs):
        """
        Transforms the data (applying normalization etc)
        and returns a tensorflow dataset.
        It also stores the meta data of features
        that is used in the reverse transformation step.

        Args:
            x: Data to transform. Must be a pandas DataFrame.

        Returns:
            Transformed data.
        """
        if not self.fitted:
            raise RuntimeError("You haven't yet fitted the DataTransformer. "
                               "You must call the `fit` method before you can call the "
                               "`transform` method. ")
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Only pandas dataframe is supported by DataTransformation class."
                             f" But data of type {type(x)} was passed. "
                             f"Please convert it to pandas dataframe before passing.")

        self.ordered_features_names_all = x.columns

        if self.categorical_features is not None:
            x[self.categorical_features] = self.encoder.transform(x[self.categorical_features])

        # we don't need to check if there are numerical features,
        # because we apply this "numerical" MinMax transformation to the whole
        # dataset, even to the categorical features we converted using ordinal encoder
        x = x.values
        x = (x - self.min_vals) / self.max_vals
        x = x.astype(np.float)

        # if return_dataframe:
        x = pd.DataFrame(x, columns=self.ordered_features_names_all)
        return x

    def reverse_transform(self,
                          x,
                          **kwargs):
        """
        Reverse Transforms the transformed data.
        The `min_vals` and `max_vals` values learnt in the `transform` step
        are used for reverse transformation of numerical features.

        Args:
            x: Transformed Data.

        Returns:
            Data in its original format and scale.
        """
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=self.ordered_features_names_all)

        if self.min_vals is None and self.max_vals is None:
            raise ValueError("The values for `min_vals` and `max_vals` do not exist. "
                             "This implies that you haven't yet used the `transform` method "
                             "to transform numerical features. ")

        # Since we apply the numerical minmax transformation to the whole dataset
        # even the ordinal encoded categorical features, so we first reverse
        # transform the numerical transformations on the whole dataset and then
        # we reverse categorical transformations if any.
        if self.numerical_features is not None:
            x = (x * self.max_vals) + self.min_vals
        if self.categorical_features is not None:
            x[self.categorical_features] = self.encoder.inverse_transform(x[self.categorical_features])

        # if not return_dataframe:
        #     x = x.values
        return x


class DataSampler:
    """
    DataSampler class that prepares the transformed data in the format
    that is expected and digestible by the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        batch_size: default 512, Batch size to use for dataset.
        shuffle: default True, Whether to shuffle the data.
        random_seed: default None, Random seed to use when shuffling.

    Example:
        ```python
        input_data = np.random.arange(100).reshape(20, 5)
        data_sampler = DataSampler()
        dataset = data_sampler.get_dataset(input_data)
        ```
    """
    def __init__(self,
                 batch_size=512,
                 shuffle=True,
                 random_seed=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.num_samples = None
        self.data_dim = None

        if self.random_seed:
            np.random.seed(self.random_seed)

    def get_dataset(self, x_transformed):
        """
        Args:
            x_transformed: `np.ndarray` or `pd.DataFrame`,
                Transformed dataset.
        Returns:
             A tensorflow dataset, that generates batches of
            size `batch_size` which is the argument passed during
            DataSampler instantiation. The dataset instance then can
            be readily passed to the GAIN fit method without requiring
            any further processing.
        """
        self.num_samples, self.data_dim = x_transformed.shape

        dataset = (tf.data.Dataset
                   .from_generator(self.generator,
                                   output_signature=(
                                       tf.TensorSpec(shape=(None, self.data_dim), dtype=tf.float32, name="x_generator"),
                                       tf.TensorSpec(shape=(None, self.data_dim), dtype=tf.float32, name="x_discriminator"),
                                        ),
                                   args=[x_transformed]
                                   )
                   )
        return dataset

    def generator(self, x):
        """
        Generator function that is used to create tensorflow dataset.
        It applies any manipulations that are required and then generates
        batches in the format required by the architecture.

        Args:
            x: Dataset to be converted into tensorflow dataset.
        """
        steps_per_epoch = self.num_samples // self.batch_size
        is_n_divisible_by_batch_size = self.num_samples % self.batch_size == 0
        steps_per_epoch += 1 if not is_n_divisible_by_batch_size else 0

        # since we need to draw a batch of data separately for generator and discriminator
        # so we generate idx separately for generator and discriminator
        # and shuffle them if necessary
        generator_idx = np.arange(self.num_samples)
        discriminator_idx = np.arange(self.num_samples)
        if self.shuffle:
            generator_idx = np.random.permutation(generator_idx)
            discriminator_idx = np.random.permutation(discriminator_idx)

        from_index = 0
        to_index = self.batch_size
        for i in range(steps_per_epoch):
            # One the last batch when the number of samples isn't divisible
            # by batch size, then trying to extract x[from_index: to_index] will
            # result in error since to_index will go out of bounds.
            # So to avoid this, we add the following check
            # and modify the to_index value accordingly.
            if i == (steps_per_epoch - 1) and not is_n_divisible_by_batch_size:
                x_generator = x[generator_idx[from_index: None]]
                x_discriminator = x[discriminator_idx[from_index: None]]
            else:
                x_generator = x[generator_idx[from_index: to_index]]
                x_discriminator = x[discriminator_idx[from_index: to_index]]
                from_index += self.batch_size
                to_index += self.batch_size

            yield x_generator, x_discriminator
