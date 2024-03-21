import tensorflow as tf
import numpy as np


class GAINDataSampler:
    """
    GAINDataSampler class that prepares the transformed data in the format
    that is expected and digestible by the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        batch_size: int, Batch size to use for dataset.
            Defaults to 512
        shuffle: bool, Whether to shuffle the data.
            Defaults to True.
        seed: int, Random seed to use when shuffling.
    """
    def __init__(self,
                 batch_size=512,
                 shuffle=True,
                 seed=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = None
        self.data_dim = None

    def get_dataset(self, x_transformed):
        """
        Args:
            x_transformed: np.ndarray or pd.DataFrame, Dataset transformed by
                GAINDataTransformer class.

        Returns:
             A tensorflow dataset, that generates batches of
            size `batch_size` which is the argument passed during
            GAINDataSampler instantiation. The dataset instance then can
            be readily passed to the GAIN fit method without requiring
            any further processing.
        """
        self.num_samples, self.data_dim = x_transformed.shape

        dataset = (tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(None, self.data_dim), dtype=tf.float32,
                              name="x_generator"),
                tf.TensorSpec(shape=(None, self.data_dim), dtype=tf.float32,
                              name="x_discriminator"),
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

        # since we need to draw a batch of data separately for generator and
        # discriminator so we generate idx separately for generator and
        # discriminator and shuffle them if necessary
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
