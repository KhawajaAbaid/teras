from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.mixture import BayesianGaussianMixture
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
                 weight_threshold=0.005,
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
            # Sample/Select a cluster for each value based on the given clusters probabilities array for that value
            selected_clusters_indices = np.apply_along_axis(
                                                    lambda c_p: np.random.choice(np.arange(len(c_p)), p=c_p),
                                                    axis=1,
                                                    arr=clusters_probs)

            # To create one-hot component, we'll store the selected clusters indices
            # and the number of valid clusters
            self.features_meta_data[feature_name]['selected_clusters_indices'] = selected_clusters_indices
            self.features_meta_data[feature_name]['num_valid_clusters'] = sum(valid_clusters_indicator)

            # Use the selected clusters to normalize the values.
            # To normalize, we need the means and standard deviations
            # Means
            clusters_means = self.bay_guass_mix.means_.sqeeuze()[valid_clusters_indicator]
            # clusters_means = clusters_means[selected_clusters_indices]
            self.features_meta_data[feature_name]['clusters_means'] = clusters_means
            # Standard Deviations
            clusters_stds = np.sqrt(self.bay_guass_mix.covariances_).squeeze()[valid_clusters_indicator]
            # clusters_stds = clusters_stds[selected_clusters_indices]
            self.features_meta_data[feature_name]['clusters_stds'] = clusters_stds

            self.fitted = True

    def transform(self, x):
        """
        Args:
            x: pandas DataFrame, TensorFlow Dataset, or numpy ndarray
        Returns:
            TBD
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
            alphas = (feature - means) / (self.std_multiplier * stds)
            alphas = np.expand_dims(alphas, 1)

            normalized_feature = np.concatenate([alphas, betas], axis=1)
            x_cont_normalized.append(normalized_feature)

        x_cont_normalized = np.asarray(x_cont_normalized).T
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
                normalized_feature = x_normalized[:, feature_name]
                x[:, feature_name] = normalized_feature * (self.std_multiplier * stds) + means
            else:
                raise ValueError(f"`x_normalized` must be either a pandas DataFrame or numpy ndarray. "
                                 f"{type(x_normalized)} was given.")
        return x


class DataTransformer:
    def __init__(self,
                 continuous_features,
                 categorical_features,
                 max_clusters=10,
                 std_multiplier=4,
                 weight_threshold=0.005,
                 covariance_type="full",
                 weight_concentration_prior_type="dirichlet_process",
                 weight_concentration_prior=0.001
                 ):
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.max_clusters = max_clusters
        self.std_multiplier = std_multiplier
        self.weight_threshold = weight_threshold
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior

        self.num_categorical_features = len(categorical_features)

        self.mode_specific_normalizer = ModeSpecificNormalization(continuous_features=self.continuous_features,
                                                             max_clusters=self.max_clusters,
                                                             std_multiplier=self.std_multiplier,
                                                             weight_threshold=self.weight_threshold,
                                                             covariance_type=self.covariance_type,
                                                             weight_concentration_prior_type=self.weight_concentration_prior_type,
                                                             weight_concentration_prior=self.weight_concentration_prior)

        self.continuous_features_meta_data = dict()
        self.categorical_features_meta_data = dict()
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
            # feature_meta_data = dict()
            num_categories = x[feature_name].nunique()
            # feature_meta_data["num_categories"] = num_categories
            num_categories_all.append(num_categories)
            # feature_meta_data["relative_index"] = feature_relative_index
            relative_indices_all.append(feature_relative_index)
            feature_relative_index += num_categories

            categories_probs_dict = ((x[feature_name].value_counts() / len(x))
                                        .apply(np.log)
                                        .to_dict())
            categories, categories_probs = categories_probs_dict.keys(), categories_probs_dict.values()
            categories_probs_all.append(categories_probs)
            categories_all.append(categories)
            # TODO: use sklearn onehot instead of pd.get_dummies, so we get not only one hotted data
            #       but also get the categories to integer labels mappings which may prove vital in
            #       reverse transformation. Store them as meta data as well.
            # self.categorical_features_meta_data[feature_name] = feature_meta_data
        self.categorical_features_meta_data["total_num_categories"] = sum(num_categories_all)
        self.categorical_features_meta_data["num_categories_all"] = num_categories_all
        self.categorical_features_meta_data["relative_indices_all"] = relative_indices_all
        self.categorical_features_meta_data["categories_probs_all"] = categories_probs_all
        self.categorical_features_meta_data["categories_all"] = categories_all

        self.one_hot_enc.fit(x)
        return self.one_hot_enc.transform(x)

    def transform(self, x):
        x_continuous = self.transform_continuous_data(x[self.continuous_features])
        self.continuous_features_meta_data = self.mode_specific_normalizer.features_meta_data
        x_categorical = self.transform_categorical_data(x[self.categorical_features])
        return np.concatenate([x_continuous, x_categorical], axis=1)

    def reverse_transform(self, x):
        # TODO
        pass


class DataSampler:
    def __init__(self, x:pd.DataFrame,
                 categorical_features=None,
                 categorical_features_meta_data=None):
        self.categorical_features = np.array(categorical_features)
        self.num_categorical_features = len(categorical_features)
        self.categorical_features_meta_data = categorical_features_meta_data

        # adapting the approach from the official implementation
        # to sample evenly across the categories to combat imbalance
        self.row_idx_by_categories = list()
        row_idx_raw = [x.groupby(feature).groups for feature in self.categorical_features]
        self.row_idx_by_categories = tf.ragged.constant([[values.to_list()
                                                          for values in feat.values()]
                                                         for feat in row_idx_raw])

    def sample_cond_vector(self, batch_size):
        # 1. Create Nd zero-filled mask vectors mi = [mi(k)] where k=1...|Di| and for i = 1,...,Nd,
        # so the ith mask vector corresponds to the ith column,
        # and each component is associated to the category of that column.
        masks = tf.zeros([batch_size, self.num_categorical_features])
        cond_vectors = tf.zeros([batch_size, self.categorical_features_meta_data["total_num_categories"]])
        # 2. Randomly select a discrete column Di out of all the Nd discrete columns, with equal probability.
        random_features_indices = tf.random.uniform([batch_size], minval=0,
                                                    maxval=self.num_categorical_features,
                                                    dtype=tf.int64).numpy()
        # random_features_names = tf.gather(self.categorical_features, random_features_indices).numpy()
        # random_feature_relative_index = self.categorical_features_meta_data[random_feature_name]["relative_index"]
        random_features_relative_indices = tf.gather(self.categorical_features_meta_data["relative_indices"],
                                                     indices=random_features_indices).numpy()
        # random_feature_num_categories = self.categorical_features_meta_data[random_feature_name]["num_categories"]
        random_features_num_categories = tf.gather(self.categorical_features_meta_data["num_categories_all"],
                                                   indices=random_features_indices).numpy()
        # 3. Construct a PMF across the range of values of the column selected in 2, Di* , such that the
        # probability mass of each value is the logarithm of its frequency in that column.
        # NOTE: We've precomputed the probabilities in the DataTransformer class for each column already
        # to speed things up.
        random_features_categories_probs = tf.gather(self.categorical_features_meta_data["categories_probs_all"],
                                                        indices=random_features_indices).numpy()
        # random_feature_categories = tf.gather(self.categorical_features_meta_data["categories_all"],
        #                                       indices=random_features_indices).numpy()
        # random_feature_probs = list(map(lambda value: random_feature_probs_dict[value], random_feature_probs_dict))
        # todo: replace the following with tf_random_choice from teras.utils
        # Choose random value index for each feature
        random_values_indices = [np.random.choice(np.arange(len(feature_probs)),
                                                  size=1,
                                                  p=feature_probs)
                                 for feature_probs in random_features_categories_probs]
        # Offset this index by relative index of the feature that it belongs to.
        # because the final cond vector is the concatenation of all features and
        # is just one vector that has the length equal to total_num_categories
        random_values_indices_offsetted = random_values_indices + random_features_relative_indices
        indices_cond = list(zip(tf.range(batch_size), random_values_indices_offsetted))
        cond_vectors = tf.tensor_scatter_nd_update(cond_vectors, indices_cond, tf.ones(batch_size))
        indices_mask = list(zip(tf.range(batch_size), random_features_indices))
        masks = tf.tensor_scatter_nd_update(masks, indices_mask, tf.ones(batch_size))
        return cond_vectors, masks, random_features_indices, random_values_indices

    def sample_data(self, feature_idx, values_idx):
        sample_idx = []
        for feat_id, val_id in zip(feature_idx, values_idx):
            sample_idx.append(np.random.choice(self.row_idx_by_categories[feat_id][val_id]))
        return self.data[sample_idx]
