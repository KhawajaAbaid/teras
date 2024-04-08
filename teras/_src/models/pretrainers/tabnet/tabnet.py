import keras

from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.models.TabNetPretrainer")
class TabNetPretrainer(backend.models.TabNetPretrainer):
    """
    TabNetPretrainer for pretraining `TabNetEncoder` as proposed by
    Arik et al. in the paper,
    "TabNet: Attentive Interpretable Tabular Learning"

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        encoder: keras.Model, instance of `TabNetEncoder` to pretrain
        decoder: keras.Model, instance of `TabNetDecoder`
        missing_feature_probability: float, probability of missing features
        seed: int, seed for generating mask. Defaults to 1337
    """
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 missing_feature_probability: float = 0.3,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            missing_feature_probability=missing_feature_probability,
            seed=seed,
            **kwargs)
