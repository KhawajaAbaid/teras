from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.mixture import BayesianGaussianMixture
from teras.utils import tf_random_choice
import numpy as np
import pandas as pd


class ModeSpecificNormalization:
    """
    Mode Specific Normalization as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        continuous_features:
            List of continuous features names.
            In the case of ndarray, pass a list of continuous column indices.
        max_clusters: Maximum clusters
        std_multiplier:
            Multiplies the standard deviation in the normalization.
            Defaults to 4 as proposed in the paper.
        weight_threshold:
            Taken from the official implementation.
            The minimum value a component weight can take to be considered a valid component.
            `weights_` under this value will be ignored.
            Defaults to 0.005.
        covariance_type: parameter for the GaussianMixtureModel class of sklearn
        weight_concentration_prior_type: parameter for the GaussianMixtureModel class of sklearn
        weight_concentration_prior: parameter for the GaussianMixtureModel class of sklearn
    """
    def __init__(self,
                 continuous_features=None,
                 max_clusters=10,
                 std_multiplier=4,
                 weight_threshold=0.,
                 covariance_type="full",
                 weight_concentration_prior_type="dirichlet_process",
                 weight_concentration_prior=0.001):
        self.continuous_features = continuous_features
        self.max_clusters = max_clusters
        self.std_multiplier = std_multiplier
        self.weight_threshold = weight_threshold
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior

        # Features meta-data dictionary will contain all the information about a feature
        # such as selected clusters indices, number of valid clusters, clusters means & stds.
        #
        # 1. `Clusters indices` will be used in the transform method
        #       to create the one hot vector B(i,j) where B stands for Beta, for each value c(i,j)
        #       where c(i,j) is the jth value in the ith continuous feature.
        #       as proposed in the paper in the steps to apply mode specific normalization method on page 3-4.
        # 2. `Number of valid clusters` will be used in the transform method when one-hotting
        # 3. `means` and `standard deviations` will be used in transform
        #       step to normalize the value c(i,j) to create a(i,j) where a stands for alpha.
        self.features_meta_data = {}

        self.bay_guass_mix = BayesianGaussianMixture(n_components=self.max_clusters,
                                                      covariance_type=self.covariance_type,
                                                      weight_concentration_prior_type=self.weight_concentration_prior_type,
                                                      weight_concentration_prior=self.weight_concentration_prior)

        self.fitted = False

    def fit(self, x):
        """
        Args:
            x: A pandas DataFrame or numpy ndarray
        """
        relative_indices_all = []
        num_valid_clusters_all = []
        relative_index = 0
        for feature_name in self.continuous_features:
            if isinstance(x, pd.DataFrame):
                feature = x[feature_name].values.reshape(-1, 1)
            elif isinstance(x, np.ndarray):
                feature = x[:, feature_name].reshape(-1, 1)
            else:
                raise ValueError(f"`x` must be either a pandas DataFrame or numpy ndarray. "
                                 f"{type(x)} was given.")

            self.features_meta_data[feature_name] = {}
            self.bay_guass_mix.fit(feature)
            # The authors use a weight threshold to filter out components in their implementation.
            # For consistency's sake, we're going to use this idea but with slight modification.
            # Reference to the official implementation: https://github.com/sdv-dev/CTGAN
            valid_clusters_indicator = self.bay_guass_mix.weights_ > self.weight_threshold
            # Compute probability of coming from each cluster for each value in the given (continuous) feature.
            clusters_probs = self.bay_guass_mix.predict_proba(feature)
            # Filter out the "invalid" clusters
            clusters_probs = clusters_probs[:, valid_clusters_indicator]
            clusters_probs[np.isnan(clusters_probs)] = 1e-10
            # Normalize probabilities to sum up to 1 for each row
            clusters_probs /= np.sum(clusters_probs, axis=1, keepdims=True)

            # Sample/Select a cluster for each value based on the given clusters probabilities array for that value
            def random_choice(probs_array):
                probs_array[np.isnan(probs_array)] = 1e-10
                probs_array /= np.sum(probs_array)
                return np.random.choice(np.arange(len(probs_array)), p=probs_array)

            selected_clusters_indices = np.apply_along_axis(
                                                    random_choice,
                                                    axis=1,
                                                    arr=clusters_probs)

            # To create one-hot component, we'll store the selected clusters indices
            # and the number of valid clusters
            num_valid_clusters = sum(valid_clusters_indicator)
            self.features_meta_data[feature_name]['selected_clusters_indices'] = selected_clusters_indices
            self.features_meta_data[feature_name]['num_valid_clusters'] = num_valid_clusters

            relative_indices_all.append(relative_index)
            # The 1 is for the alpha feature, since each continuous feature
            # gets transformed into alpha + betas where,
            # |alpha| = 1 and |betas| = num_valid_clusters
            relative_index += (1 + num_valid_clusters)
            num_valid_clusters_all.append(num_valid_clusters)

            # Use the selected clusters to normalize the values.
            # To normalize, we need the means and standard deviations
            # Means
            clusters_means = self.bay_guass_mix.means_.squeeze()[valid_clusters_indicator]
            self.features_meta_data[feature_name]['clusters_means'] = clusters_means
            # Standard Deviations
            clusters_stds = np.sqrt(self.bay_guass_mix.covariances_).squeeze()[valid_clusters_indicator]
            self.features_meta_data[feature_name]['clusters_stds'] = clusters_stds

        self.features_meta_data["relative_indices_all"] = np.array(relative_indices_all)
        self.features_meta_data["num_valid_clusters_all"] = num_valid_clusters_all
        self.fitted = True

    def transform(self, x):
        """
        Args:
            x: A pandas DataFrame
        Returns:
            A numpy ndarray of transformed continuous data
        """
        # Contain the normalized continuous features
        x_cont_normalized = []
        for feature_name in self.continuous_features:
            selected_clusters_indices = self.features_meta_data[feature_name]['selected_clusters_indices']
            num_valid_clusters = self.features_meta_data[feature_name]['num_valid_clusters']
            # One hot components for all values in the feature
            # we borrow the beta notation from the paper
            # for clarity and understanding's sake.
            betas = np.eye(num_valid_clusters)[selected_clusters_indices]

            # Normalizing
            means = self.features_meta_data[feature_name]['clusters_means']
            stds = self.features_meta_data[feature_name]['clusters_stds']
            if isinstance(x, pd.DataFrame):
                feature = x[feature_name].values
            elif isinstance(x, np.ndarray):
                feature = x[:, feature_name]
            else:
                raise ValueError(f"`x` must be either a pandas DataFrame or numpy ndarray. "
                                 f"{type(x)} was given.")
            means = means[selected_clusters_indices]
            stds = stds[selected_clusters_indices]
            alphas = (feature - means) / (self.std_multiplier * stds)
            alphas = np.expand_dims(alphas, 1)

            normalized_feature = np.concatenate([alphas, betas], axis=1)
            x_cont_normalized.append(normalized_feature)

        x_cont_normalized = np.concatenate(x_cont_normalized, axis=1)
        return x_cont_normalized

    def fit_transform(self, x):
        """
        Fits and transforms x.
        Returns:
            A normalized copy of x.
        """
        self.fit(x)
        return self.transform(x)

    def reverse_transform(self, x_normalized):
        """
        Args:
            x: DataFrame or n-dimensional numpy array
        Returns:
            Convert the normalized features to original values,
            effectively reversing the transformation
        """
        x = x_normalized.copy()
        for feature_name in self.continuous_features:
            means = self.features_clusters_means[feature_name]
            stds = self.features_clusters_stds[feature_name]
            if isinstance(x_normalized, pd.DataFrame):
                normalized_feature = x_normalized[feature_name].values
                x[feature_name] = normalized_feature * (self.std_multiplier * stds) + means
            elif isinstance(x_normalized, np.ndarray):
                # todo delete this part and just stick with pandas dataframe
                normalized_feature = x_normalized[:, feature_name]
                x[:, feature_name] = normalized_feature * (self.std_multiplier * stds) + means
            else:
                raise ValueError(f"`x_normalized` must be either a pandas DataFrame or numpy ndarray. "
                                 f"{type(x_normalized)} was given.")
        return x


class DataTransformer:
    """
    Data Transformation class based on the data transformation
    in the official CTGAN paper and implementation.

    Reference(s):
        https://arxiv.org/abs/1907.00503
        https://github.com/sdv-dev/CTGAN/

    Args:
        continuous_features: List of continuous features names
        categorical_features: List of categorical features names
        max_clusters: Maximum Number of clusters to use in ModeSpecificNormalization
            Defaults to 10
        std_multiplier:
            Multiplies the standard deviation in the normalization.
            Defaults to 4 as proposed in the paper.
        weight_threshold:
            Taken from the official implementation.
            The minimum value a component weight can take to be considered a valid component.
            `weights_` under this value will be ignored.
            Defaults to 0.005
        covariance_type: parameter for the GaussianMixtureModel class of sklearn
            Defaults to 'full'
        weight_concentration_prior_type: parameter for the GaussianMixtureModel class of sklearn
            Defaults to 'dirichlet_process'
        weight_concentration_prior: parameter for the GaussianMixtureModel class of sklearn.
            Defaults to 0.001
    """
    def __init__(self,
                 continuous_features=None,
                 categorical_features=None,
                 max_clusters=10,
                 std_multiplier=4,
                 weight_threshold=0.005,
                 covariance_type="full",
                 weight_concentration_prior_type="dirichlet_process",
                 weight_concentration_prior=0.001
                 ):
        self.continuous_features = continuous_features if continuous_features else []
        self.categorical_features = categorical_features if categorical_features else []
        self.max_clusters = max_clusters
        self.std_multiplier = std_multiplier
        self.weight_threshold = weight_threshold
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior

        self.num_categorical_features = len(categorical_features)
        self.num_continuous_features = len(continuous_features)

        self.mode_specific_normalizer = None
        if self.num_continuous_features > 0:
            self.mode_specific_normalizer = ModeSpecificNormalization(
                                                    continuous_features=self.continuous_features,
                                                    max_clusters=self.max_clusters,
                                                    std_multiplier=self.std_multiplier,
                                                    weight_threshold=self.weight_threshold,
                                                    covariance_type=self.covariance_type,
                                                    weight_concentration_prior_type=self.weight_concentration_prior_type,
                                                    weight_concentration_prior=self.weight_concentration_prior)
        self.features_meta_data = dict()
        self.categorical_values_probs = dict()
        self.one_hot_enc = OneHotEncoder()

    def transform_continuous_data(self, x):
        return self.mode_specific_normalizer.fit_transform(x)

    def transform_categorical_data(self, x):
        # To speedup computation of conditional vector down the road,
        # we assign a relative index to each feature. For instance,
        # Given three categorical columns Gender(2 categories), City (4 categories) and EconomicClass (5 categories)
        # Relative indexes will be calculated as below:
        #   gender_relative_index: 0
        #   city_relative_index: gender_relative_index + num_categories_in_gender => 0 + 2 = 2
        #   economic_class_relative_index: city_relative_index + num_categories_in_city => 2 + 4 = 6

        categorical_features_meta_data = dict()
        # NOTE: The purpose of using these lists is that, this way we'll later be able to access
        # metadata for multiple features at once using their indices rather than names
        # which would've been required in case of a dict and would have been less efficient
        relative_indices_all = []
        num_categories_all = []
        # For every feature we'll compute the probabilities over the range of values based on their freqs
        # and then append that probabilities array to the following mother array
        categories_probs_all = []
        # A nested list where each element corresponds to the list of categories in the feature
        categories_all = []
        feature_relative_index = 0
        for feature_name in self.categorical_features:
            num_categories = x[feature_name].nunique()
            num_categories_all.append(num_categories)
            relative_indices_all.append(feature_relative_index)
            feature_relative_index += num_categories

            log_freqs = x[feature_name].value_counts().apply(np.log)
            categories_probs_dict = log_freqs.to_dict()
            # To overcome the floating point precision issue which causes probabilities to not sum up to 1
            # and resultantly causes error in np.random.choice method,
            # we round the probabilities to 7 decimal points
            probs = np.around(np.array(list(categories_probs_dict.values())), 7)
            # Normalizing so all probs sum up to 1
            probs = probs / np.sum(probs)
            categories, categories_probs = list(categories_probs_dict.keys()), probs
            categories_probs_all.append(categories_probs)
            categories_all.append(categories)
        categorical_features_meta_data["total_num_categories"] = sum(num_categories_all)
        categorical_features_meta_data["num_categories_all"] = num_categories_all
        categorical_features_meta_data["relative_indices_all"] = np.array(relative_indices_all)
        categorical_features_meta_data["categories_probs_all"] = categories_probs_all
        categorical_features_meta_data["categories_all"] = categories_all

        self.features_meta_data["categorical"] = categorical_features_meta_data

        self.one_hot_enc.fit(x)
        return self.one_hot_enc.transform(x)

    def transform(self, x):
        total_transformed_features = 0
        x_continuous, x_categorical = None, None
        if self.num_continuous_features > 0:
            x_continuous = self.transform_continuous_data(x[self.continuous_features])
            self.features_meta_data["continuous"] = self.mode_specific_normalizer.features_meta_data
            total_transformed_features += (self.features_meta_data["continuous"]["relative_indices_all"][-1] +
                                            self.features_meta_data["continuous"]["num_valid_clusters_all"][-1])
        if self.num_categorical_features > 0:
            x_categorical = self.transform_categorical_data(x[self.categorical_features])
            total_transformed_features += (self.features_meta_data["categorical"]["relative_indices_all"][-1] +
                                            self.features_meta_data["categorical"]["num_categories_all"][-1] + 1)

        # since we concatenate the categorical features AFTER the continuous alphas and betas
        # so we'll create an overall relative indices array where we offset the relative indices
        # of the categorical features by the total number of continuous features components
        relative_indices_all = []
        offset = 0
        if x_continuous is not None:
            cont_relative_indices = self.features_meta_data["continuous"]["relative_indices_all"]
            relative_indices_all.extend(cont_relative_indices)
            offset = cont_relative_indices[-1] + self.features_meta_data["continuous"]["num_valid_clusters_all"][-1]
        if x_categorical is not None:
            # +1 since the categorical relative indices start at 0
            relative_indices_all.extend(self.features_meta_data["categorical"]["relative_indices_all"] + 1 + offset)
        self.features_meta_data["relative_indices_all"] = relative_indices_all
        self.features_meta_data["total_transformed_features"] = total_transformed_features
        x_transformed = np.concatenate([x_continuous, x_categorical.toarray()], axis=1)
        return x_transformed

    @property
    def categorical_features_meta_data(self):
        """
        Returns:
            A dictionary of categorical features meta data
        """
        if self.num_categorical_features == 0:
            return None
        return self.features_meta_data["categorical"]

    @property
    def continuous_features_meta_data(self):
        """
        Returns:
            A dictionary of continuous features meta data
        """
        if self.num_continuous_features == 0:
            return None
        return self.features_meta_data["continuous"]

    def reverse_transform(self, x_generated):
        """
        Reverses transforms the generated data to the original data format.

        Args:
            x_generated: Generated dataset.

        Returns:
            Generated data in the original data format.
        """
        all_features = self.continuous_features + self.categorical_features
        if self.num_continuous_features > 0:
            num_valid_clusters_all = self.continuous_features_meta_data["num_valid_clusters_all"]
        if self.num_categorical_features > 0:
            num_categories_all = self.categorical_features_meta_data["num_categories_all"]
        data = {}
        cat_index = 0       # categorical index
        cont_index = 0      # continuous index
        for index, feature_name in enumerate(all_features):
            # the first n features are continuous
            if index < len(self.continuous_features):
                alphas = x_generated[index]
                betas = x_generated[index + 1 : index + 1 + num_valid_clusters_all[cont_index]]
                # Recall that betas represent the one hot encoded form of the cluster number
                cluster_indices = np.argmax(betas, axis=1)
                # List of cluster means for a feature. contains one value per cluster
                means = self.continuous_features_meta_data[feature_name]["cluster_means"]

                # Since each individual element within the cluster is associated with
                # one of the cluster's mean. We use the `cluster_indices` to get
                # a list of size len(x) where each element is a mean for the corresponding
                # element in the feature
                means = means[cluster_indices]
                # Do the same for stds
                stds = self.continuous_features_meta_data[feature_name]["cluster_stds"]
                stds = stds[cluster_indices]
                feature = alphas * (self.std_multiplier * stds) + means
                data[feature_name] = feature
                cont_index += 1
            else:
                # if the index is greater than or equal to len(continuous_features),
                # then the column at hand is categorical
                raw_feature_parts = x_generated[index: index + num_categories_all[cat_index]]
                categories = self.one_hot_enc.categories_[cat_index]
                categories_indices = tf.argmax(raw_feature_parts, axis=1)
                feature = categories[categories_indices]
                data[feature_name] = feature
                cat_index += 1


class DataSampler:
    """
    Data Sampler class based on the data sampler class
    in the official CTGAN implementation.

    Reference(s):
        https://arxiv.org/abs/1907.00503
        https://github.com/sdv-dev/CTGAN/

    Args:
        x_original: Original Dataset - a pandas DataFrame
        x_transformed: Dataset transformed using DataTransformer class
        categorical_features: List of categorical features names
        categorical_features_meta_data: A dict of meta data for categorical features.
            You can access that from the DataTransformer.categorical_features_meta_data
    """
    def __init__(self,
                 x_original: pd.DataFrame,
                 x_transformed = None,
                 categorical_features=None,
                 categorical_features_meta_data=None):
        self.x_original = x_original
        self.x_transformed = x_transformed
        self.categorical_features = np.array(categorical_features)
        self.num_categorical_features = len(categorical_features)
        self.categorical_features_meta_data = categorical_features_meta_data

        self.num_samples = len(x_original)

        # adapting the approach from the official implementation
        # to sample evenly across the categories to combat imbalance
        self.row_idx_by_categories = list()
        row_idx_raw = [x_original.groupby(feature).groups for feature in self.categorical_features]
        self.row_idx_by_categories = tf.ragged.constant([[values.to_list()
                                                          for values in feat.values()]
                                                         for feat in row_idx_raw])

    def get_dataset(self, batch_size):
        """
        Returns a tensoflow dataset that utilizes the sample_data method
        to create batches of data. This way user can just pass the dataset object to the fit
        method of the model and each batch generated will satisfies all out requirements of sampling
        """
        self.batch_size = batch_size
        dataset = tf.data.Dataset.from_generator(
            self.sample_data,
            output_signature=(tf.TensorSpec(shape=(batch_size, tf.shape(self.x_transformed)[1]), name="data_batch"),
                              tf.TensorSpec(shape=(batch_size,), dtype=tf.int32, name="shuffled_idx"),
                              tf.TensorSpec(shape=(batch_size,), dtype=tf.int32, name="random_features_indices"),
                              tf.TensorSpec(shape=(batch_size,), dtype=tf.int32, name="random_values_indices"))
        )
        return dataset

    def sample_cond_vector_for_training(self, batch_size,
                                        random_features_indices=None,
                                        random_values_indices=None):
        # 1. Create Nd zero-filled mask vectors mi = [mi(k)] where k=1...|Di| and for i = 1,...,Nd,
        # so the ith mask vector corresponds to the ith column,
        # and each component is associated to the category of that column.
        masks = tf.zeros([batch_size, self.num_categorical_features])
        cond_vectors = tf.zeros([batch_size, self.categorical_features_meta_data["total_num_categories"]])
        # 2. Randomly select a discrete column Di out of all the Nd discrete columns, with equal probability.
        # >>> We select them in sample_data method

        random_features_relative_indices = tf.gather(self.categorical_features_meta_data["relative_indices_all"],
                                                     indices=random_features_indices)

        # 3. Construct a PMF across the range of values of the column selected in 2, Di* , such that the
        # probability mass of each value is the logarithm of its frequency in that column.
        # >>> Moved to the sample_data method, which runs before this method
        # and populates the value of `self.random_features_indices`

        # Choose random value index for each feature
        # >>> Moved to the sample_data method, which runs before this method
        # and populates the value of `self.random_values_indices`

        # Offset this index by relative index of the feature that it belongs to.
        # because the final cond vector is the concatenation of all features and
        # is just one vector that has the length equal to total_num_categories
        random_values_indices_offsetted = random_values_indices + tf.cast(random_features_relative_indices,
                                                                          dtype=tf.int32)
        # Indices are required by the tensor_scatter_nd_update method to be of type int32 and NOT int64
        # random_values_indices_offsetted = tf.cast(random_values_indices_offsetted, dtype=tf.int32)
        # indices_cond = list(zip(tf.range(batch_size), random_values_indices_offsetted))
        indices_cond = tf.stack([tf.range(batch_size), random_values_indices_offsetted], axis=1)
        ones = tf.ones(batch_size)
        cond_vectors = tf.tensor_scatter_nd_update(cond_vectors, indices_cond, ones)
        # indices_mask = list(zip(tf.range(batch_size), tf.cast(random_features_indices, dtype=tf.int32)))
        indices_mask = tf.stack([tf.range(batch_size), random_features_indices], axis=1)
        masks = tf.tensor_scatter_nd_update(masks, indices_mask, tf.ones(batch_size))
        return cond_vectors, masks

    def sample_cond_vector_for_generation(self, batch_size):
        """
        The difference between this method and the training one is that, here
        we sample indices purely randomly instead of based on the calculated
        probability as proposed in the paper.
        """
        if self.num_categorical_features == 0:
            return None
        num_categories_all = self.categorical_features_meta_data["num_categories_all"]
        cond_vectors = tf.zeros((batch_size, self.categorical_features_meta_data["total_num_categories"]))
        random_features_indices = tf.random.uniform(shape=(batch_size,),
                                                    minval=0, maxval=self.num_categorical_features,
                                                    dtype=tf.int32)

        # For each randomly picked feature, we get it's corresponding num_categories
        random_num_categories_all = tf.gather(num_categories_all,
                                              indices=random_features_indices)
        # Then we select one category index from a feature using a range of 0 — num_categories
        random_values_indices = [tf.random.uniform(shape=(1,),
                                    minval=0, maxval=num_categories, dtype=tf.int32)
                                 for num_categories in random_num_categories_all]
        # Offset this index by relative index of the feature that it belongs to.
        # because the final cond vector is the concatenation of all features and
        # is just one vector that has the length equal to total_num_categories
        random_features_relative_indices = tf.gather(self.categorical_features_meta_data["relative_indices_all"],
                                                     indices=random_features_indices)
        random_values_indices_offsetted = tf.constant(random_values_indices) + random_features_relative_indices
        random_values_indices_offsetted = tf.cast(random_values_indices_offsetted, dtype=tf.int32)
        indices_cond = list(zip(tf.range(batch_size), random_values_indices_offsetted))
        ones = tf.ones(batch_size)
        cond_vectors = tf.tensor_scatter_nd_update(cond_vectors, indices_cond, ones)
        return cond_vectors

    def sample_data(self):
        """
        Used to create a tensorflow dataset.
        Returns:
            A batch of data, shuffled_idx
        """
        # This random_feature_indices variable is required during the sample_cond vector method
        # but since we're using sample_data function to create out tensorflow dataset, this
        # gets called first to generate a batch, so keep in mind that this is where this
        # variable gets its values. We could alternatively just return these indices
        # and pass them as argument to the sample cond_vec but for now let's just work with it.
        num_steps_per_epoch = self.num_samples // self.batch_size
        for _ in range(num_steps_per_epoch):
            random_features_indices = tf.random.uniform([self.batch_size], minval=0,
                                                        maxval=self.num_categorical_features,
                                                        dtype=tf.int64)
            # NOTE: We've precomputed the probabilities in the DataTransformer class for each feature already
            # to speed things up.
            random_features_categories_probs = tf.gather(tf.ragged.constant(self.categorical_features_meta_data["categories_probs_all"]),
                                                            indices=random_features_indices)
            random_values_indices = [tf_random_choice(np.arange(len(feature_probs)),
                                                      n_samples=1,
                                                      p=feature_probs)
                                     for feature_probs in random_features_categories_probs]
            random_values_indices = tf.squeeze(random_values_indices)
            # the official implementation uses actual indices during the sample_cond_vector method
            # but uses the shuffled version in sampling data, so we're gonna do just that.
            shuffled_idx = tf.random.shuffle(tf.range(self.batch_size))
            features_idx = random_features_indices
            values_idx = random_values_indices
            if shuffled_idx is not None:
                features_idx = tf.gather(features_idx, indices=shuffled_idx)
                values_idx = tf.gather(values_idx, indices=shuffled_idx)
            sample_idx = []
            for feat_id, val_id in zip(features_idx, values_idx):
                s_id = tf_random_choice(self.row_idx_by_categories[tf.squeeze(feat_id)][tf.squeeze(val_id)],
                                                   n_samples=1)
                sample_idx.append(tf.squeeze(s_id))
            # we also return shuffled_idx because it will be required to shuffle the conditional vector
            # in the training loop as we want to keep the shuffling consistent as the batch of
            # transformed data and cond vector must have one to one feature correspondence.
            yield self.x_transformed[sample_idx], shuffled_idx, random_features_indices, random_values_indices