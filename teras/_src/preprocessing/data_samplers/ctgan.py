import numpy as np

from teras._src.typing import FeaturesNamesType

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
        batch_size: int, default 512,
            Batch size to use for the dataset.
        seed: int, Seed for random ops.
    """
    def __init__(self,
                 metadata,
                 categorical_features: FeaturesNamesType,
                 continuous_features: FeaturesNamesType = None,
                 batch_size: int = 512,
                 seed: int = 1337):
        self.metadata = metadata
        self.batch_size = batch_size
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.seed = seed

        self._np_rng = np.random.default_rng(self.seed)
        self.num_samples = None
        self.data_dim = None
        self.batch_size = batch_size
        self.row_idx_by_categories = list()
        self._num_categorical_features = len(self.categorical_features)
        self._cat_features_relative_idx = np.asarray(
            self.metadata["categorical"]["relative_indices_all"])
        self._total_categories = self.metadata["categorical"]["total_num_categories"]
        self._all_categories = self.metadata["categorical"]["categories_all"]
        self._categories_probs_all = self.metadata["categorical"]["categories_probs_all"]

        # Since the nested lists have different lengths, so lets pad
        max_num_categories = max([len(categories)
                                  for categories in self._all_categories])
        self._features_categories_probs = np.array([
            np.pad(probs, (0, max_num_categories - len(probs)),
                   constant_values=0.)
            for probs in self._categories_probs_all
        ])

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
            output_signature=((
                tf.TensorSpec(
                    shape=(None, self.data_dim),
                    name="real_samples"),
                tf.TensorSpec(
                    shape=(None, total_num_categories),
                    dtype=tf.float32, name="cond_vectors_real"),
                tf.TensorSpec(
                    shape=(None, total_num_categories,),
                    dtype=tf.float32, name="cond_vectors"),
                tf.TensorSpec(
                    shape=(None, self._num_categorical_features,),
                    dtype=tf.float32, name="mask")
            ),
                tf.TensorSpec(shape=(None, 1),
                              dtype=tf.float32, name="dummy_vals"),
            ),
            args=(x_transformed,)
        )
        return dataset

    def sample_cond_vectors_for_training(self, batch_size):
        # 1. Create Nd zero-filled mask vectors mi = [mi(k)] where k=1...|Di|
        # and for i = 1,...,Nd, so the ith mask vector corresponds to the ith
        # column, and each component is associated to the category of that
        # column.
        mask = np.zeros((batch_size, self._num_categorical_features))
        cond_vectors = np.zeros((batch_size, self._total_categories))
        # 2. Randomly select a discrete column Di out of all the Nd discrete
        # columns, with equal probability.
        selected_cat_features_idx = self._np_rng.choice(
            np.arange(self._num_categorical_features),
            size=batch_size
        )
        selected_cat_features_relative_idx = self._cat_features_relative_idx[
            selected_cat_features_idx]

        # 3. Construct a PMF across the range of values of the column selected
        # in 2, Di* , such that the probability mass of each value is the
        # logarithm of its frequency in that column.

        # NOTE: We've precomputed the probabilities in the DataTransformer
        # class for each feature already to speed things up.
        selected_features_categories_probs = self._features_categories_probs[
            selected_cat_features_idx]

        # Choose random values idx for features
        selected_cat_values_idx = np.array([
            self._np_rng.choice(np.arange(len(probs)),
                                p=probs)
            for probs in selected_features_categories_probs]
        ).astype(np.int32)

        # Offset this index by relative index of the feature that it belongs to.
        # because the final cond vector is the concatenation of all features and
        # is just one vector that has the length equal to total_num_categories
        selected_cat_values_idx_offsetted = selected_cat_features_relative_idx

        cond_vectors[np.arange(batch_size), selected_cat_values_idx_offsetted] = 1
        mask[np.arange(batch_size), selected_cat_features_idx] = 1
        return (cond_vectors, mask, selected_cat_features_idx,
                selected_cat_values_idx)

    def sample_cond_vectors_for_generation(self, batch_size):
        """
        The difference between this method and the training one is that, here
        we sample indices purely randomly instead of based on the calculated
        probability as proposed in the paper.
        """
        num_categories_all = np.array(
            self.metadata["categorical"]["num_categories_all"])
        cond_vectors = np.zeros((batch_size, self._total_categories))
        selected_cat_features_idx = self._np_rng.choice(
            np.arange(self._num_categorical_features),
            size=batch_size
        )
        selected_cat_features_relative_idx = self._cat_features_relative_idx[
            selected_cat_features_idx]

        # For each randomly picked feature, we get it's corresponding
        # num_categories
        selected_num_categories_all = num_categories_all[selected_cat_features_idx]
        # Then we select one category index from a feature using a range of
        # 0 — num_categories
        selected_values_idx = np.array([
            self._np_rng.choice(np.arange(num_categories))
            for num_categories in selected_num_categories_all]
        ).astype(np.int32)
        # Offset this index by relative index of the feature that it belongs to.
        # because the final cond vector is the concatenation of all features and
        # is just one vector that has the length equal to total_num_categories
        selected_values_idx += selected_cat_features_relative_idx
        cond_vectors[np.arange(batch_size), selected_values_idx] = 1
        return cond_vectors

    def generator(self, x_transformed):
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
            # `cond_vectors` will be first concatenated with the noise
            # vector `z` to create generator input and then will be concatenated
            # with the generated samples to serve as input for discriminator
            (
                cond_vectors, mask, selected_cat_features_idx,
                selected_cat_values_idx
            ) = self.sample_cond_vectors_for_training(self.batch_size)

            # the official implementation uses actual indices during the
            # sample_cond_vector method but uses the shuffled version in
            # sampling data, so we're gonna do just that.
            shuffled_idx = np.arange(self.batch_size)
            self._np_rng.shuffle(shuffled_idx)

            shuffled_cat_features_idx = selected_cat_features_idx[shuffled_idx]
            shuffled_values_idx = selected_cat_values_idx[shuffled_idx]

            sample_idx = []
            for feat_id, val_id in zip(shuffled_cat_features_idx,
                                       shuffled_values_idx):
                s_id = self._np_rng.choice(
                    self.row_idx_by_categories[np.squeeze(feat_id)][
                        np.squeeze(val_id)])
                sample_idx.append(np.squeeze(s_id))

            # we also return shuffled_idx because it will be required to shuffle
            # the conditional vector in the training loop as we want to keep
            # the shuffling consistent as the batch of transformed data and
            # cond vector must have one to one feature correspondence.

            # `cond_vectors_real` will be concatenated with the real_samples
            # and passed to the discriminator
            cond_vectors_real = cond_vectors[shuffled_idx]
            real_samples = x_transformed[sample_idx]
            dummy_ys = np.ones((self.batch_size, 1))
            yield (real_samples, cond_vectors_real, cond_vectors, mask), dummy_ys
