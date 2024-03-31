from keras import ops
from keras import random
from teras.utils.types import FeaturesNamesType
import numpy as np
try:
    import tensorflow as tf
except:
    raise ImportError(
        "You need tensorflow to use CTGANDataSampler. "
        "Install it using `pip install tensorflow`"
    )
import tensorflow as tf


class CTGANDataSampler:
    """
    CTGANDataSampler class based on the data sampler class
    in the official CTGAN implementation.

    Reference(s):
        https://arxiv.org/abs/1907.00503
        https://github.com/sdv-dev/CTGAN/

    Args:
        metadata: dict, A dictionary of metadata computed during data
            transformation. You can access it from the ``.get_metadata()`` of
            ``CTGANDataTransformer`` instance.
        categorical_features: list, List of categorical features names.
            CTGAN requires dataset to have at least one categorical feature,
            if your dataset doesn't contain any categorical features,
            consider using some other generative model.
        continuous_features: list, List of continuous features names
        batch_size: ``int``, default 512,
            Batch size to use for the dataset.
    """
    def __init__(self,
                 metadata,
                 categorical_features: FeaturesNamesType,
                 continuous_features: FeaturesNamesType = None,
                 batch_size: int = 512):
        self.metadata = metadata
        self.batch_size = batch_size
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        self.num_samples = None
        self.data_dim = None
        self.batch_size = batch_size
        self.row_idx_by_categories = list()

    def get_dataset(self,
                    x_transformed,
                    x_original):
        """
        Args:
            x_transformed: Dataset transformed using DataTransformer class
            x_original: Original Dataset - a pandas DataFrame.
                It is used for computing categorical values' probabilities
                for later sampling.
        Returns:
            Returns a tensorflow dataset that utilizes the sample_data method
            to create batches of data. This way user can just pass the dataset
            object to the fit method of the model and each batch generated
            will satisfy all out requirements of sampling
        """
        self.num_samples, self.data_dim = x_transformed.shape
        # adapting the approach from the official implementation
        # to sample evenly across the categories to combat imbalance
        row_idx_raw = [x_original.groupby(feature).groups
                       for feature in self.categorical_features]
        self.row_idx_by_categories = [
            [values.to_list() for values in feat.values()]
            for feat in row_idx_raw]

        total_num_categories = self.metadata["categorical"]["total_num_categories"]

        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(tf.TensorSpec(shape=(self.batch_size, tf.shape(x_transformed)[1]),
                                            name="real_samples"),
                              tf.TensorSpec(shape=(self.batch_size, total_num_categories),
                                            dtype=tf.float32, name="cond_vectors_real"),
                              tf.TensorSpec(shape=(self.batch_size, total_num_categories),
                                            dtype=tf.float32, name="cond_vectors"),
                              tf.TensorSpec(shape=(self.batch_size, len(self.categorical_features)),
                                            dtype=tf.float32, name="mask")),
            args=[x_transformed]
        )
        return dataset

    def sample_cond_vectors_for_training(self,
                                         random_features_idx=None,
                                         random_values_idx=None):
        # 1. Create Nd zero-filled mask vectors mi = [mi(k)] where k=1...|Di|
        # and for i = 1,...,Nd, so the ith mask vector corresponds to the ith
        # column, and each component is associated to the category of that
        # column.
        masks = ops.zeros([self.batch_size, len(self.categorical_features)])
        cond_vectors = ops.zeros(
            [self.batch_size, self.metadata["categorical"]["total_num_categories"]])
        # 2. Randomly select a discrete column Di out of all the Nd discrete
        # columns, with equal probability.
        # >>> We select them in generator method

        random_features_relative_indices = ops.take(
            self.metadata["categorical"]["relative_indices_all"],
            indices=random_features_idx)

        # 3. Construct a PMF across the range of values of the column selected
        # in 2, Di* , such that the probability mass of each value is the
        # logarithm of its frequency in that column.
        # >>> Moved to the generator method, which calls this method
        # and passes the value of `random_features_idx`

        # Choose random value index for each feature
        # >>> Moved to the generator method, which calls this method
        # and passes the value of `random_values_idx`

        # Offset this index by relative index of the feature that it belongs to.
        # because the final cond vector is the concatenation of all features and
        # is just one vector that has the length equal to total_num_categories
        random_values_idx_offsetted = random_values_idx + ops.cast(
            random_features_relative_indices, dtype="int32")
        # Indices are required by the tensor_scatter_nd_update method to be of
        # type int32 and NOT int64 random_values_idx_offsetted = tf.cast(
        # random_values_idx_offsetted, dtype=tf.int32)
        # indices_cond = list(zip(tf.range(batch_size), random_values_idx_offsetted))
        indices_cond = ops.stack([ops.arange(self.batch_size), random_values_idx_offsetted], axis=1)
        ones = ops.ones(self.batch_size)
        cond_vectors = ops.scatter_update(cond_vectors, indices_cond, ones)
        # indices_mask = list(zip(tf.range(batch_size), tf.cast(random_features_idx, dtype=tf.int32)))
        indices_mask = ops.stack([ops.arange(self.batch_size),
                                  random_features_idx], axis=1)
        masks = ops.scatter_update(masks, indices_mask,
                                   ops.ones(self.batch_size))
        return cond_vectors, masks

    def sample_cond_vectors_for_generation(self, batch_size):
        """
        The difference between this method and the training one is that, here
        we sample indices purely randomly instead of based on the calculated
        probability as proposed in the paper.
        """
        num_categories_all = self.metadata["categorical"]["num_categories_all"]
        cond_vectors = ops.zeros(
            (batch_size, self.metadata["categorical"]["total_num_categories"]))
        random_features_idx = ops.cast(
            random.uniform(shape=(batch_size,),
                           minval=0,
                           maxval=len(self.categorical_features)),
            dtype="int32")

        # For each randomly picked feature, we get it's corresponding
        # num_categories
        random_num_categories_all = ops.take(num_categories_all,
                                             indices=random_features_idx)
        # Then we select one category index from a feature using a range of
        # 0 — num_categories
        random_values_idx = [ops.squeeze(ops.cast(
            random.uniform(shape=(1,),
                           minval=0,
                           maxval=num_categories),
            dtype="int32"))
            for num_categories in random_num_categories_all]
        random_values_idx = ops.stack(random_values_idx)
        # Offset this index by relative index of the feature that it belongs to.
        # because the final cond vector is the concatenation of all features and
        # is just one vector that has the length equal to total_num_categories
        random_features_relative_indices = ops.take(
            self.metadata["categorical"]["relative_indices_all"],
            indices=random_features_idx)
        random_values_idx_offsetted = random_values_idx + ops.cast(
            random_features_relative_indices, dtype="int32")
        # random_values_idx_offsetted = tf.cast(random_values_idx_offsetted, dtype=tf.int32)
        indices_cond = list(zip(ops.arange(batch_size),
                                random_values_idx_offsetted))
        ones = ops.ones(batch_size)
        cond_vectors = ops.scatter_update(cond_vectors, indices_cond, ones)
        return cond_vectors

    def generator(self, x_transformed, for_tvae=False):
        """
        Used to create a tensorflow dataset.
        Returns:
            A batch of data
        """
        # This random_feature_indices variable is required during the
        # sample_cond vector method but since we're using sample_data
        # function to create out tensorflow dataset, this gets called first
        # to generate a batch, so keep in mind that this is where this
        # variable gets its values. We could alternatively just return these
        # indices and pass them as argument to the sample cond_vec but for
        # now let's just work with it.
        num_steps_per_epoch = self.num_samples // self.batch_size
        for _ in range(num_steps_per_epoch):
            random_features_idx = ops.cast(
                random.uniform([self.batch_size],
                               minval=0,
                               maxval=len(self.categorical_features)),
                dtype="int32")
            # NOTE: We've precomputed the probabilities in the DataTransformer
            # class for each feature already to speed things up.
            random_features_categories_probs = ops.take(
                self.metadata["categorical"]["categories_probs_all"],
                indices=random_features_idx)
            random_values_idx = [
                random_choice(feature_probs)
                for feature_probs in random_features_categories_probs]
            random_values_idx = ops.cast(ops.squeeze(random_values_idx),
                                         dtype="int32")
            # the official implementation uses actual indices during the
            # sample_cond_vector method ut uses the shuffled version in
            # sampling data, so we're gonna do just that.
            shuffled_idx = random.shuffle(ops.arange(self.batch_size))
            # features_idx = random_features_idx
            # values_idx = random_values_idx
            shuffled_features_idx = ops.take(random_features_idx,
                                             indices=shuffled_idx)
            shuffled_values_idx = ops.take(random_values_idx,
                                           indices=shuffled_idx)
            sample_idx = []
            for feat_id, val_id in zip(shuffled_features_idx,
                                       shuffled_values_idx):
                s_id = np.random.choice(
                    self.row_idx_by_categories[ops.squeeze(feat_id)][ops.squeeze(val_id)])
                sample_idx.append(ops.squeeze(s_id))

            # we also return shuffled_idx because it will be required to shuffle
            # the conditional vector in the training loop as we want to keep
            # the shuffling consistent as the batch of transformed data and
            # cond vector must have one to one feature correspondence.
            # yield x_transformed[sample_idx], shuffled_idx,
            # random_features_idx, random_values_idx

            # `cond_vectors` will be first concatenated with the noise
            # vector `z` to create generator input and then will be concatenated
            # with the generated samples to serve as input for discriminator
            cond_vectors, mask = self.sample_cond_vectors_for_training(
                random_features_idx=random_features_idx,
                random_values_idx=random_values_idx)
            # `cond_vectors_real` will be concatenated with the real_samples
            # and passed to the discriminator
            cond_vectors_real = ops.take(cond_vectors, indices=shuffled_idx)
            real_samples = x_transformed[sample_idx]
            yield real_samples, cond_vectors_real, cond_vectors, mask
