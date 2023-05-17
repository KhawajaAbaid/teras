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
        """
        Args:

        """
        self.continuous_features = continuous_features
        self.max_clusters = max_clusters
        self.std_multiplier = std_multiplier
        self.weight_threshold = weight_threshold
        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior

        self.features_clusters_means = {}
        self.features_clusters_stds = {}
        self.bay_guass_mix = BayesianGaussianMixture(n_components=self.max_clusters,
                                                      covariance_type=self.covariance_type,
                                                      weight_concentration_prior_type=self.weight_concentration_prior_type,
                                                      weight_concentration_prior=self.weight_concentration_prior)

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
            selected_clusters = np.apply_along_axis(
                                                    lambda c_p: np.random.choice(np.arange(len(c_p)), p=c_p),
                                                    axis=1,
                                                    arr=clusters_probs)

            # Use the selected clusters to normalize the values.
            # To normalize, we need the means and standard deviations
            # Means
            clusters_means = self.bay_guass_mix.means_.sqeeuze()[valid_clusters_indicator]
            clusters_means = clusters_means[selected_clusters]
            self.features_clusters_means[feature_name] = clusters_means
            # Standard Deviations
            clusters_stds = np.sqrt(self.bay_guass_mix.covariances_).squeeze()[valid_clusters_indicator]
            clusters_stds = clusters_stds[selected_clusters]
            self.features_clusters_stds[feature_name] = clusters_stds

    def transform(self, x):
        """
        Args:
            x: DataFrame or n-dimensional numpy array
        Returns:
            A normalized copy of x.
        """
        x_normalized = x.copy()
        for feature_name in self.continuous_features:
            means = self.features_clusters_means[feature_name]
            stds = self.features_clusters_stds[feature_name]
            if isinstance(x, pd.DataFrame):
                feature = x[feature_name].values
                x_normalized[feature_name] = (feature - means) / (self.std_multiplier * stds)
            elif isinstance(x, np.ndarray):
                feature = x[:, feature_name]
                x_normalized[:, feature_name] = (feature - means) / (self.std_multiplier * stds)
            else:
                raise ValueError(f"`x` must be either a pandas DataFrame or numpy ndarray. "
                                 f"{type(x)} was given.")
        return x_normalized

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
