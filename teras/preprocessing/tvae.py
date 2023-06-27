import tensorflow as tf
import numpy as np
from teras.preprocessing.ctgan import (DataTransformer,
                                       DataSampler as _BaseDataSampler)
from teras.utils import tf_random_choice


class DataSampler(_BaseDataSampler):
    """
    DataSampler class for TVAE that subclass the DataSampler class
    from CTGAN.
    The two classes share much functionality since TVAE and CTGAN
    are proposed in the same paper and almost all preprocessing
    for both is same.
    There are, however, are a few differences in the `get_dataset`
    and `generator` method, hence this new subclassed class.

    Args:
        batch_size: `int`, default 512, Batch size to use for the dataset.
        categorical_features: List of categorical features names
        numerical_features: List of numerical features names
        meta_data: Namedtuple of features meta data computed during data transformation.
            You can access it from the `.get_meta_data()` of DataTransformer instance.
    """
    def __init__(self,
                 batch_size=512,
                 categorical_features=None,
                 numerical_features=None,
                 meta_data=None
                 ):
        super().__init__(batch_size=batch_size,
                         categorical_features=categorical_features,
                         numerical_features=numerical_features,
                         meta_data=meta_data)

    def get_dataset(self,
                    x_transformed,
                    x_original=None):
        """
        Args:
            x_transformed: Dataset transformed using DataTransformer class
            x_original: Original Dataset - a pandas DataFrame.
                It is used for computing categorical values' probabilities
                for later sampling.
        Returns:
            Returns a tensorflow dataset that utilizes the sample_data method
            to create batches of data. This way user can just pass the dataset object to the fit
            method of the model and each batch generated will satisfies all out requirements of sampling
        """
        self.num_samples, self.data_dim = x_transformed.shape
        # adapting the approach from the official implementation
        # to sample evenly across the categories to combat imbalance
        row_idx_raw = [x_original.groupby(feature).groups for feature in self.categorical_features]
        self.row_idx_by_categories = tf.ragged.constant([[values.to_list()
                                                          for values in feat.values()]
                                                         for feat in row_idx_raw])

        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(tf.TensorSpec(shape=(self.batch_size, tf.shape(x_transformed)[1]), name="data_batch")
                              ),
            args=[x_transformed],
        )
        return dataset

    def generator(self, x_transformed):
        """
        Used to create a tensorflow dataset.
        Returns:
            A batch of data
        """
        # This random_feature_indices variable is required during the sample_cond vector method
        # but since we're using sample_data function to create out tensorflow dataset, this
        # gets called first to generate a batch, so keep in mind that this is where this
        # variable gets its values. We could alternatively just return these indices
        # and pass them as argument to the sample cond_vec but for now let's just work with it.
        num_steps_per_epoch = self.num_samples // self.batch_size
        for _ in range(num_steps_per_epoch):
            random_features_idx = tf.random.uniform([self.batch_size], minval=0,
                                                    maxval=len(self.categorical_features),
                                                    dtype=tf.int32)
            # NOTE: We've precomputed the probabilities in the DataTransformer class for each feature already
            # to speed things up.
            random_features_categories_probs = tf.gather(tf.ragged.constant(self.meta_data.categorical.categories_probs_all),
                                                         indices=random_features_idx)
            random_values_idx = [tf_random_choice(np.arange(len(feature_probs)),
                                                  n_samples=1,
                                                  p=feature_probs)
                                 for feature_probs in random_features_categories_probs]
            random_values_idx = tf.cast(tf.squeeze(random_values_idx), dtype=tf.int32)
            # the official implementation uses actual indices during the sample_cond_vector method
            # but uses the shuffled version in sampling data, so we're gonna do just that.
            shuffled_idx = tf.random.shuffle(tf.range(self.batch_size))
            shuffled_features_idx = tf.gather(random_features_idx, indices=shuffled_idx)
            shuffled_values_idx = tf.gather(random_values_idx, indices=shuffled_idx)
            sample_idx = []
            # TODO make it more efficient -- maybe replace with gather or gathernd or something
            for feat_id, val_id in zip(shuffled_features_idx, shuffled_values_idx):
                s_id = tf_random_choice(self.row_idx_by_categories[tf.squeeze(feat_id)][tf.squeeze(val_id)],
                                        n_samples=1)
                sample_idx.append(tf.squeeze(s_id))

            yield x_transformed[sample_idx]
