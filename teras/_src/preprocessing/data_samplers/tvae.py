try:
    import tensorflow as tf
except:
    raise ImportError(
        "You need tensorflow to use `TVAEataSampler`. "
        "Install it using `pip install tensorflow`"
    )
import numpy as np
from teras._src.preprocessing.data_samplers.ctgan import CTGANDataSampler as _BaseDataSampler
from teras._src.typing import FeaturesNamesType


class TVAEDataSampler(_BaseDataSampler):
    """
    TVAEDataSampler class for `TVAE` architecture.
    It subclasses the `CTGANDataSampler` class from `CTGAN` architecture.

    The two classes share much functionality since `TVAE` and `CTGAN`
    are proposed in the same paper and almost all preprocessing
    for both is same.
    There are, however, are a few differences in the `get_dataset`
    and `generator` methods, hence this new subclassed class.

    Reference(s):
        https://arxiv.org/abs/1907.00503
        https://github.com/sdv-dev/CTGAN/

    Args:
        metadata: dict, A dictionary of metadata computed during data
            transformation. You can access it from the `.get_metadata()` of
            `TVAEDataTransformer` instance.
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
                 metadata: dict,
                 categorical_features: FeaturesNamesType = None,
                 continuous_features: FeaturesNamesType = None,
                 batch_size: int = 512,
                 seed: int = 1337,
                 ):
        super().__init__(metadata=metadata,
                         categorical_features=categorical_features,
                         continuous_features=continuous_features,
                         batch_size=batch_size,
                         seed=seed)

    def get_dataset(self,
                    x_transformed,
                    x_original=None):
        """
        Args:
            x_transformed: Dataset transformed using `TVAEDataTransformer` class
            x_original: Original Dataset - a pandas DataFrame.
                It is used for computing categorical values' probabilities
                for later sampling.
        Returns:
            Returns a tensorflow dataset that utilizes the `generator` method
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

        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.batch_size, tf.shape(x_transformed)[1]),
                    name="data_batch")
                              ),
            args=(x_transformed,),
        )
        return dataset

    def generator(self, x_transformed):
        """
        Used to create a tensorflow dataset.

        Args:
            x_transformed: Dataset transformed by the `TVAEDataTransformer`
                class.

        Returns:
            A batch of data
        """
        num_steps_per_epoch = self.num_samples // self.batch_size
        for _ in range(num_steps_per_epoch):
            selected_cat_features_idx = self._np_rng.choice(
                np.arange(self._num_categorical_features),
                size=self.batch_size
            )
            # NOTE: We've precomputed the probabilities in the DataTransformer
            # class for each feature already to speed things up.
            selected_features_categories_probs = self._features_categories_probs[selected_cat_features_idx]

            # Choose random values idx for features
            selected_cat_values_idx = np.array([
                self._np_rng.choice(np.arange(len(probs)),
                                    p=probs)
                for probs in selected_features_categories_probs]
            ).astype(np.int32)

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

            yield x_transformed[sample_idx]
