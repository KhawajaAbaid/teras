import tensorflow as tf
import numpy as np
import pandas as pd
from teras.preprocessing.gain import DataSampler as GainDataSampler, DataTransformer


class DataSampler(GainDataSampler):
    """
    DataSampler class for PC-GAIN.
    It extends GAIN's DataSampler class because PC-GAIN
    in itself is an extension of GAIN architecture,
    so the data required by them is in similar format.
    But PC-GAIN requires one extra thing -- the pretraining dataset.
    Hence, the need to extend the GAIN Data Sampler class and
    implement that functionality here.

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        batch_size: Batch size to use for dataset.
            Defaults to 512.
        shuffle: Whether to shuffle the dataset.
            Defaults to True.
        random_seed: Random seed to make shuffling
            and any other random operations consistent.
    """
    def __init__(self,
                 batch_size=512,
                 shuffle=True,
                 random_seed=None):
        super().__init__(batch_size=batch_size,
                         shuffle=shuffle,
                         random_seed=random_seed)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed

    def get_pretraining_dataset(self,
                                x_transformed,
                                pretraining_size=0.2,
                                batch_size=None):
        """
        Args:
            x_transformed: Transformed dataset.
            pretraining_size: Fraction of dataset that will be used for pretraining.
                As a general rule of thumb (based on official implementation),
                    - for datasets with `num_samples >= 20000`, use `pretraining_size=0.2`
                    - for datasets with `num_samples < 20000`, use `pretraining_size = 0.4`
                Please note that, this is NOT a hard and fast rule. You can specify
                whatever fraction size you deem reasonable for your use case.
            batch_size: In case you want to use a different batch size for your
                pretraining than for actual training. If None, the batch_size
                specified while DataSample instantiation will be used.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Sort all the samples according to their missing rate and select the top
        # lambda a.k.a. `pretraining_size` samples to make the pretraining dataset.
        # Missing rate in simple terms means,
        # the number of nans per row divided by the total number of element in the row.
        # Or in other terms, number of nans per row divided by the dimensionality of dataset.
        if isinstance(x_transformed, pd.DataFrame):
            x_transformed = x_transformed.values
        num_samples, dim = x_transformed.shape
        num_pretraining_samples = int(pretraining_size * num_samples)
        missing_rate_per_row = np.sum(np.isnan(x_transformed), axis=1) / dim
        # Note that the below indices will be in ascending order
        sorted_idx = np.argsort(missing_rate_per_row)
        # Since sorted idx are in ascending order and we need `pretraining_size` fraction
        # of samples with least missing rate. so we select the first `num_pretraining_samples` idx
        pretraining_samples_idx = sorted_idx[:num_pretraining_samples]

        # x_transformed but with only pretraining samples
        x_transformed = x_transformed[pretraining_samples_idx]

        # We need separate batches of data for generator and discriminator on each step as required by
        # GAIN and hence PC-GAIN
        gen_idx = np.random.permutation(np.arange(len(x_transformed)))
        disc_idx = np.random.permutation(np.arange(len(x_transformed)))

        pretraining_dataset = tf.data.Dataset.from_generator(
            generator=self._pretraining_dataset_generator,
            output_signature=(tf.TensorSpec(shape=(None, dim), dtype=tf.float32, name="generator_pretraining_data"),
                              tf.TensorSpec(shape=(None, dim), dtype=tf.float32, name="discriminator_pretraining_data")
                              ),
            args=[x_transformed, gen_idx, disc_idx, batch_size]
        )
        return pretraining_dataset

    def _pretraining_dataset_generator(self, x, gen_idx, disc_idx, batch_size):
        """
         Generator function that is used to create pretraining tesnorflow dataset.
         It applies any manipulations that are required and then generates
         batches in the format required by the architecture.
         """
        num_samples, dim = x.shape
        steps_per_epoch = num_samples // self.batch_size
        is_n_divisible_by_batch_size = num_samples % self.batch_size == 0
        steps_per_epoch += 1 if not is_n_divisible_by_batch_size else 0
        from_index = 0
        to_index = batch_size
        for i in range(steps_per_epoch):
            # On the last batch when the number of samples isn't divisible
            # by batch size, then trying to extract x[from_index: to_index] will
            # result in error since to_index will go out of bounds.
            # So to avoid this, we add the following check
            # and modify the to_index value accordingly.
            if i == (steps_per_epoch - 1) and not is_n_divisible_by_batch_size:
                x_generator = x[gen_idx[from_index: None]]
                x_discriminator = x[disc_idx[from_index: None]]
            else:
                x_generator = x[gen_idx[from_index: to_index]]
                x_discriminator = x[disc_idx[from_index: to_index]]
                from_index += batch_size
                to_index += batch_size

            yield x_generator, x_discriminator
