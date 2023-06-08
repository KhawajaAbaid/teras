import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import OrdinalEncoder


FEATURE_NAMES_TYPE = List[str]


class DataTransformer:
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
        normalize: Whether to use MinMax normalization at all.
            It is strongly recommended to use normalization.

    Example:
        ```python
            input_df = pd.DataFrame(np.arange(10, 3).reshape(20, 5),
                                    columns=['A', 'B', 'C'])
            numerical_features = input_df.columns
            data_transformer = DataTransformer(categorical_features=None,
                                               numerical_features=numerical_features,
                                               normalize=True)
            transformed_df = data_transformer.transform(input_df)
        ```
    """
    def __init__(self,
                 categorical_features: FEATURE_NAMES_TYPE = None,
                 numerical_features: FEATURE_NAMES_TYPE = None,
                 normalize=True):
        if categorical_features is None and numerical_features is None:
            raise ValueError("You must pass features names list for at least one of "
                             "categorical or numerical types, if not both.")
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.all_features_names = categorical_features + numerical_features
        self.normalize = normalize
        self.min_vals = []
        self.max_vals = []

    def transform_categorical(self, x: pd.DataFrame):
        """
        Applies ordinal encoding to categorical features

        x: Transformed data with numerical features.
            Must be a pandas DataFrame.

        Returns:
            Encoded categorical data.
        """
        self.oe = OrdinalEncoder()
        return self.oe.fit_transform(x[self.categorical_features])

    def reverse_transform_categorical(self, x: pd.DataFrame):
        """
        Rounds values and reverses the transformation to original values.

        Args:
            x: Data with categorical features that is to be transformed.
                Must be a pandas DataFrame.

        Returns:
            Categorical data in its original format.
        """
        return self.oe.inverse_transform(x[self.categorical_features].round().astype(np.float))

    def transform_numerical(self, x):
        """
        Applies the MinMax transformation in accordance with the official
        GAIN paper and implementation.
        We don't just normalize numerical features values but all values,
        even the categorical features values that we converted using ordinal encoder.

        Returns:
            Transformed numerical data.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        if self.normalize:
            self.min_vals = np.nanmin(x, axis=0)
            self.max_vals = np.nanmax(x, axis=0)
            x = (x - self.min_vals) / self.max_vals
            x = x.astype(np.float)
        return x

    def reverse_transform_numerical(self, x):
        """
        Reverses the numerical transformation.

        Args:
            x: Transformed data

        Returns:
            Numerical data in its original format.
        """
        if self.min_vals is None and self.max_vals is None:
            raise ValueError("The values for `self.min_vals` and `self.max_vals` are not yet set. "
                             "This implies that you haven't yet used the `transform` method "
                             "to transform numerical features. ")
        x = (x * self.max_vals) + self.min_vals
        return x

    def transform(self,
                  x: pd.DataFrame,
                  return_dataframe=False):
        """
        Transforms the data (applying normalization etc)
        and returns a tensorflow dataset.
        It also stores the meta data of features
        that is used in the reverse transformation step.

        Args:
            x: Data to transform. Must be a pandas DataFrame.
            return_dataframe: default False, whether to return data as pandas dataframe.

        Returns:
            Transformed data.
        """
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Only pandas dataframe is supported by DataTransformation class."
                             f" But data of type {type(x)} was passed. "
                             f"Please convert it to pandas dataframe before passing.")

        self.all_ordered_features_names = x.columns

        if self.categorical_features is not None:
            x[self.categorical_features] = self.transform_categorical(x)

        # we don't need to check if there are numerical features,
        # because we apply this "numerical" MinMax transformation to the whole
        # dataset, even to the categorical features we converted using ordinal encoder
        x = self.transform_numerical(x)

        if return_dataframe:
            x = pd.DataFrame(x, columns=self.all_ordered_features_names)
        return x

    def reverse_transform(self,
                          x,
                          return_dataframe=True):
        """
        Reverse Transforms the transformed data.
        The `min_vals` and `max_vals` values learnt in the `transform` step
        are used for reverse transformation of numerical features.

        Args:
            x: Transformed Data.
            return_dataframe: default True, whether to return data
                as pandas dataframe.

        Returns:
            Data in its original format and scale.
        """
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=self.all_ordered_features_names)

        # Since we apply the numerical minmax transformation to the whole dataset
        # even the ordinal encoded categorical features, so we first reverse
        # transform the numerical transformations on the whole dataset and then
        # we reverse categorical transformations if any.
        if self.numerical_features is not None:
            x = self.reverse_transform_numerical(x)
        if self.categorical_features is not None:
            x[self.categorical_features] = self.reverse_transform_categorical(x)

        if return_dataframe:
            x = pd.DataFrame(x, columns=self.all_ordered_features_names)
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

    def get_dataset(self, x_transformed):
        """
        Args:
            Transformed data.
        Returns:
             A tensorflow dataset, that generates batches of
            size `batch_size` which is the argument passing during
            DataSampler instantiation. The dataset instance then can
            be readily passed to the GAIN fit method without requiring
            any further processing.
        """
        num_samples, dim = x_transformed.shape
        generator_idx = np.arange(num_samples)
        discriminator_idx = np.arange(num_samples)
        if self.shuffle:
            if self.random_seed:
                np.random.seed(self.random_seed)
            # since we need to draw a batch of data separately for generator and discriminator
            # so we generate shuffled idx for separately for generator and discriminator
            gen_idx = np.random.permutation(generator_idx)
            disc_idx = np.random.permutation(discriminator_idx)

        dataset = (tf.data.Dataset
                   .from_generator(self.generator,
                                   output_signature=(
                                       tf.TensorSpec(shape=(None, dim), dtype=tf.float32, name="x_generator"),
                                       tf.TensorSpec(shape=(None, dim), dtype=tf.float32, name="x_discriminator"),
                                        ),
                                   args=[x_transformed, generator_idx, discriminator_idx]
                                   )
                   )
        return dataset

    def generator(self, x, generator_idx, discriminator_idx):
        """
        Generator function that is used to create tensorflow dataset.
        It applies any manipulations that are required and then generates
        batches in the format required by the architecture.

        Args:
            x: Dataset to be converted into tensorflow dataset.
            generator_idx: A numpy array of indices for
                generating data batches for generator.
            discriminator_idx: A numpy array of indices for
                generating data batches for discriminator.
        """
        num_samples, dim = x.shape
        steps_per_epoch = num_samples // self.batch_size
        is_n_divisible_by_batch_size = num_samples % self.batch_size == 0
        steps_per_epoch += 1 if not is_n_divisible_by_batch_size else 0
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
