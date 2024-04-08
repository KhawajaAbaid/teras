from teras._src.preprocessing.data_transformers.ctgan import CTGANDataTransformer as _BaseDataTransformer
from teras._src.typing import FeaturesNamesType


class TVAEDataTransformer(_BaseDataTransformer):
    """
    TVAEDataTransformer class that is exactly similar to the
    CTGANDataTransformer, it just acts as a wrapper for convenience.

    Reference(s):
        https://arxiv.org/abs/1907.00503
        https://github.com/sdv-dev/CTGAN/

    Args:
        categorical_features: list, List of categorical features names in the
            dataset.
        continuous_features: list, List of continuous features names in the
            dataset.
        max_clusters: int, Maximum Number of clusters to use in
            `ModeSpecificNormalization`. Defaults to 10.
        std_multiplier: int, Multiplies the standard deviation in the
            normalization. Defaults to 4.
        weight_threshold: float, The minimum value a component weight can
            take to be considered a valid component. `weights_` under this
            value will be ignored. (Taken from the official implementation.)
            Defaults to 0.005.
        covariance_type: str, Parameter for the `GaussianMixtureModel`
            class of sklearn. Defaults to "full".
        weight_concentration_prior_type: str, Parameter for the
            `GaussianMixtureModel` class of sklearn. Defaults to
            "dirichlet_process"
        weight_concentration_prior: float, Parameter for the
            `GaussianMixtureModel` class of sklearn. Defaults to 0.001.
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
        super().__init__(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            max_clusters=max_clusters,
            std_multiplier=std_multiplier,
            weight_threshold=weight_threshold,
            covariance_type=covariance_type,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior)
