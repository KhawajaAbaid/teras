from keras import ops
from keras import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
from teras.preprocessing.data_transformers.data_transformer import DataTransformer
from teras.utils.types import FeaturesNamesType
import concurrent.futures
import numpy as np


# Sample/Select a cluster for each value based on the given clusters
# probabilities array for that value
def random_choice(probs_array):
    probs_array[np.isnan(probs_array)] = 1e-10
    probs_array /= np.sum(probs_array)
    return np.random.choice(np.arange(len(probs_array)), p=probs_array)


def continuous_feature_transformer(args):
    x, feature_name, weight_threshold, bgm_kwargs = args
    print("processing feature: ", feature_name)
    if isinstance(x, pd.DataFrame):
        feature = x[feature_name].values.reshape(-1, 1)
    elif isinstance(x, np.ndarray):
        feature = x[:, feature_name].reshape(-1, 1)
    else:
        raise ValueError(
            f"`x` must be either a pandas DataFrame or numpy ndarray. "
            f"{type(x)} was given.")

    feature_metadata = {}
    feature_metadata["name"] = feature_name
    bay_guass_mix = BayesianGaussianMixture(**bgm_kwargs)
    bay_guass_mix.fit(feature)
    # The authors use a weight threshold to filter out components in their
    # implementation.
    # For consistency's sake, we're going to use this idea but with slight
    # modification.
    # Reference to the official implementation:
    # https://github.com/sdv-dev/CTGAN
    valid_clusters_indicator = bay_guass_mix.weights_ > weight_threshold
    # Compute probability of coming from each cluster for each value in the
    # given (continuous) feature.
    clusters_probs = bay_guass_mix.predict_proba(feature)
    # Filter out the "invalid" clusters
    clusters_probs = clusters_probs[:, valid_clusters_indicator]
    clusters_probs[np.isnan(clusters_probs)] = 1e-10
    # Normalize probabilities to sum up to 1 for each row
    clusters_probs /= np.sum(clusters_probs, axis=1, keepdims=True)
    selected_clusters_indices = np.apply_along_axis(
        random_choice,
        axis=1,
        arr=clusters_probs)
    # To create one-hot component, we'll store the selected clusters indices
    # and the number of valid clusters
    num_valid_clusters = sum(valid_clusters_indicator)
    feature_metadata[
        'selected_clusters_indices'] = selected_clusters_indices
    feature_metadata["num_valid_clusters"] = num_valid_clusters
    # Use the selected clusters to normalize the values.
    # To normalize, we need the means and standard deviations
    # Means
    clusters_means = bay_guass_mix.means_.squeeze()[
        valid_clusters_indicator]
    feature_metadata['clusters_means'] = clusters_means
    # Standard Deviations
    clusters_stds = np.sqrt(bay_guass_mix.covariances_).squeeze()[
        valid_clusters_indicator]
    feature_metadata['clusters_stds'] = clusters_stds
    return feature_metadata


class ModeSpecificNormalization:
    """
    Mode Specific Normalization as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        continuous_features: ``List[str]``
            List of continuous features names.
            In the case of ndarray, pass a list of continuous column indices

        max_clusters: ``int``, default 10,
            Maximum clusters

        std_multiplier: ``int``, default 4,
            Multiplies the standard deviation in the normalization.
            Defaults to 4 as proposed in the paper.

        weight_threshold: ``int``, default 0.005,
            Taken from the official implementation.
            The minimum value a component weight can take to be considered
            a valid component.
            `weights_` under this value will be ignored.

        covariance_type: ``str``,
            Parameter for the ``GaussianMixtureModel`` class of sklearn

        weight_concentration_prior_type: ``str``,
            default "dirichlet_process",
            Parameter for the ``GaussianMixtureModel`` class of sklearn

        weight_concentration_prior: ``float``, default 0.01,
            Parameter for the GaussianMixtureModel class of sklearn
    """
    def __init__(self,
                 continuous_features: FeaturesNamesType = None,
                 max_clusters: int = 10,
                 std_multiplier: int = 4,
                 weight_threshold: float = 0.,
                 covariance_type: str = "full",
                 weight_concentration_prior_type: str = "dirichlet_process",
                 weight_concentration_prior: float = 0.001):
        self.continuous_features = continuous_features
        self.max_clusters = max_clusters
        self.std_multiplier = std_multiplier
        self.weight_threshold = weight_threshold
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior

        # Features meta-data dictionary will contain all the information
        # about a feature such as selected clusters indices, number of
        # valid clusters, clusters means & stds.
        # 1. `Clusters indices` will be used in the transform method
        #       to create the one hot vector B(i,j) where B stands for
        #       Beta, for each value c(i,j) where c(i,j) is the jth
        #       value in the ith continuous feature as proposed in the
        #       paper in the steps to apply mode specific normalization
        #       method on page 3-4.
        # 2. `Number of valid clusters` will be used in the transform
        #       method when one-hotting
        # 3. `means` and `standard deviations` will be used in transform
        #       step to normalize the value c(i,j) to create a(i,j) where
        #       a stands for alpha.
        self.metadata = {}

        self.bay_guass_mix = BayesianGaussianMixture(
            n_components=self.max_clusters,
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

        feat_args = [(x, feature_name, self.weight_threshold,
                      self.bgm_kwargs)
                     for feature_name in self.continuous_features]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(continuous_feature_transformer,
                                   feat_args)
            for r in results:
                _name = r["name"]
                relative_indices_all.append(relative_index)
                relative_index += (1 + r["num_valid_clusters"])
                num_valid_clusters_all.append(r["num_valid_clusters"])
                del r["name"]
                self.metadata[_name] = r
        self.metadata["relative_indices_all"] = np.array(relative_indices_all)
        self.metadata["num_valid_clusters_all"] = num_valid_clusters_all
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
            selected_clusters_indices = self.metadata[feature_name][
                'selected_clusters_indices']
            num_valid_clusters = self.metadata[feature_name][
                'num_valid_clusters']
            # One hot components for all values in the feature
            # we borrow the beta notation from the paper
            # for clarity and understanding's sake.
            betas = ops.eye(num_valid_clusters)[selected_clusters_indices]

            # Normalizing
            means = self.metadata[feature_name]['clusters_means']
            stds = self.metadata[feature_name]['clusters_stds']
            if isinstance(x, pd.DataFrame):
                feature = x[feature_name].values
            elif isinstance(x, ops.ndarray):
                feature = x[:, feature_name]
            else:
                raise ValueError(f"`x` must be either a pandas DataFrame "
                                 f"or numpy ndarray. {type(x)} was given.")
            means = means[selected_clusters_indices]
            stds = stds[selected_clusters_indices]
            alphas = (feature - means) / (self.std_multiplier * stds)
            alphas = ops.expand_dims(alphas, 1)

            normalized_feature = ops.concatenate([alphas, betas], axis=1)
            x_cont_normalized.append(normalized_feature)

        x_cont_normalized = ops.concatenate(x_cont_normalized, axis=1)
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
            elif isinstance(x_normalized, ops.ndarray):
                # todo delete this part and just stick with pandas dataframe
                normalized_feature = x_normalized[:, feature_name]
                x[:, feature_name] = normalized_feature * (self.std_multiplier * stds) + means
            else:
                raise ValueError(f"`x_normalized` must be either a pandas DataFrame or numpy ndarray. "
                                 f"{type(x_normalized)} was given.")
        return x


class CTGANDataTransformer(DataTransformer):
    """
    Data Transformation class based on the data transformation
    in the official CTGAN paper and implementation.

    Reference(s):
        https://arxiv.org/abs/1907.00503
        https://github.com/sdv-dev/CTGAN/

    Args:
        categorical_features: ``List[str]``,
            List of categorical features names in the dataset.

        continuous_features: ``List[str]``,
            List of continuous features names in the dataset.

        max_clusters: ``int``, default 10,
            Maximum Number of clusters to use in ``ModeSpecificNormalization``

        std_multiplier: ``int``, default 4,
            Multiplies the standard deviation in the normalization.

        weight_threshold: ``float``, default 0.005,
            The minimum value a component weight can take to be considered a valid component.
            `weights_` under this value will be ignored.
            (Taken from the official implementation.)

        covariance_type: ``str``, default "full",
            Parameter for the ``GaussianMixtureModel`` class of sklearn.

        weight_concentration_prior_type: ``str``, default "dirichlet_process",
            Parameter for the ``GaussianMixtureModel`` class of sklearn

        weight_concentration_prior: ``float``, default 0.001,
            Parameter for the ``GaussianMixtureModel`` class of sklearn.
    """
    def __init__(self,
                 continuous_features: FeaturesNamesType = None,
                 categorical_features: FeaturesNamesType = None,
                 max_clusters: int = 10,
                 std_multiplier: int = 4,
                 weight_threshold: float = 0.005,
                 covariance_type: str = "full",
                 weight_concentration_prior_type: str = "dirichlet_process",
                 weight_concentration_prior: float = 0.001
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
        self.categorical_values_probs = dict()
        self.one_hot_enc = OneHotEncoder()
        self.metadata = dict()
        self.metadata["categorical"] = dict()
        self.metadata["continuous"] = dict()

    def transform_continuous_data(self, x):
        return self.mode_specific_normalizer.fit_transform(x)

    def transform_categorical_data(self, x):
        # To speedup computation of conditional vector down the road,
        # we assign a relative index to each feature. For instance,
        # Given three categorical columns Gender(2 categories),
        # City (4 categories) and EconomicClass (5 categories)
        # Relative indexes will be calculated as below:
        #   gender_relative_index: 0
        #   city_relative_index: gender_relative_index + num_categories_in_gender => 0 + 2 = 2
        #   economic_class_relative_index: city_relative_index + num_categories_in_city => 2 + 4 = 6

        categorical_metadata = dict()
        # NOTE: The purpose of using these lists is that, this way we'll later
        # be able to access metadata for multiple features at once using
        # their indices rather than names which would've been required in
        # case of a dict and would have been less efficient?
        relative_indices_all = []
        num_categories_all = []
        # For every feature we'll compute the probabilities over the range of
        # values based on their freqs and then append that probabilities
        # array to the following mother array
        categories_probs_all = []
        # A nested list where each element corresponds to the list of
        # categories in the feature
        categories_all = []
        feature_relative_index = 0
        for feature_name in self.categorical_features:
            num_categories = x[feature_name].nunique()
            num_categories_all.append(num_categories)
            relative_indices_all.append(feature_relative_index)
            feature_relative_index += num_categories

            log_freqs = x[feature_name].value_counts().apply(ops.log)
            categories_probs_dict = log_freqs.to_dict()
            # To overcome the floating point precision issue which causes
            # probabilities to not sum up to 1 and resultantly causes error
            # in ops.random.choice method, we round the probabilities to 7
            # decimal points
            probs = ops.round(ops.array(list(categories_probs_dict.values())),
                              decimals=7)
            # Normalizing so all probs sum up to 1
            probs = probs / ops.sum(probs)
            categories, categories_probs = list(categories_probs_dict.keys()), probs
            categories_probs_all.append(categories_probs)
            categories_all.append(categories)
        categorical_metadata["total_num_categories"] = sum(num_categories_all)
        categorical_metadata["num_categories_all"] = num_categories_all
        categorical_metadata["relative_indices_all"] = ops.array(
            relative_indices_all)
        categorical_metadata["categories_probs_all"] = categories_probs_all
        categorical_metadata["categories_all"] = categories_all

        self.metadata["categorical"] = categorical_metadata

        self.one_hot_enc.fit(x)
        return self.one_hot_enc.transform(x)

    def fit(self, x, **kwargs):
        # we've got nothing to fit in this case
        pass

    def transform(self, x):
        total_transformed_features = 0
        x_continuous, x_categorical = None, None
        if self.num_continuous_features > 0:
            x_continuous = self.transform_continuous_data(
                x[self.continuous_features])
            self.metadata["continuous"] = self.mode_specific_normalizer.metadata
            total_transformed_features += (
                    self.metadata["continuous"]["relative_indices_all"][-1] +
                    self.metadata["continuous"]["num_valid_clusters_all"][-1])
        if self.num_categorical_features > 0:
            x_categorical = self.transform_categorical_data(
                x[self.categorical_features])
            total_transformed_features += (
                    self.metadata["categorical"]["relative_indices_all"][-1] +
                    self.metadata["categorical"]["num_categories_all"][-1] + 1)

        # since we concatenate the categorical features AFTER the continuous
        # alphas and betas so we'll create an overall relative indices array
        # where we offset the relative indices of the categorical features by
        # the total number of continuous features components
        relative_indices_all = []
        offset = 0
        if x_continuous is not None:
            cont_relative_indices = self.metadata["continuous"]["relative_indices_all"]
            relative_indices_all.extend(cont_relative_indices)
            offset = cont_relative_indices[-1] + self.metadata["continuous"]["num_valid_clusters_all"][-1]
        if x_categorical is not None:
            # +1 since the categorical relative indices start at 0
            relative_indices_all.extend(self.metadata["categorical"]["relative_indices_all"] + 1 + offset)
        self.metadata["relative_indices_all"] = relative_indices_all
        self.metadata["total_transformed_features"] = total_transformed_features
        x_transformed = ops.concatenate([x_continuous, x_categorical.toarray()], axis=1)
        return x_transformed

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
            num_valid_clusters_all = self.metadata["continuous"]["num_valid_clusters_all"]
        if self.num_categorical_features > 0:
            num_categories_all = self.metadata["categorical"]["num_categories_all"]
        data = {}
        cat_index = 0       # categorical index
        cont_index = 0      # continuous index
        for index, feature_name in enumerate(all_features):
            # the first n features are continuous
            if index < len(self.continuous_features):
                alphas = x_generated[:, index]
                betas = x_generated[:, index + 1 : index + 1 + num_valid_clusters_all[cont_index]]
                # Recall that betas represent the one hot encoded form of the
                # cluster number
                cluster_indices = ops.argmax(betas, axis=1)
                # List of cluster means for a feature. contains one value per
                # cluster
                means = self.metadata["continuous"][feature_name]["clusters_means"]

                # Since each individual element within the cluster is associated
                # with one of the cluster's mean. We use the
                # `cluster_indices` to get a list of size len(x) where each
                # element is a mean for the corresponding element in the feature
                means = means[cluster_indices]
                # Do the same for stds
                stds = self.metadata["continuous"][feature_name]["clusters_stds"]
                stds = stds[cluster_indices]
                feature = alphas * (self.std_multiplier * stds) + means
                data[feature_name] = feature
                cont_index += 1
            else:
                # if the index is greater than or equal to
                # len(continuous_features), then the column at hand is
                # categorical
                raw_feature_parts = x_generated[:, index: index + num_categories_all[cat_index]]
                categories = self.one_hot_enc.categories_[cat_index]
                categories_indices = ops.argmax(raw_feature_parts, axis=1)
                feature = categories[categories_indices]
                data[feature_name] = feature
                cat_index += 1
        return pd.DataFrame(data)
