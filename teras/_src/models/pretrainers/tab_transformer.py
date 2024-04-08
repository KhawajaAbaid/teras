import keras

from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.models.TabTransformerMLMPretrainer")
class TabTransformerMLMPretrainer(backend.models.TabTransformerMLMPretrainer):
    """
    Masked Language Modelling (MLM) based Pretrainer for pretraining
    `TabTransformerBackbone` as proposed by Huang et al. in the paper,
    "TabTransformer: Tabular Data Modeling Using Contextual Embeddings".

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        model: keras.Model, instance of `TabTransformerBackbone` to
            pretrain
        data_dim: int, dimensionality of the input dataset
        missing_rate: float, fraction of original features to make missing.
            Must be in the range [0, 1).
            Defaults to 0.3 (or 30%)
        mask_seed: int, seed for generating mask.
            Defaults to 1337
    """
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 missing_rate: float = 0.3,
                 mask_seed: int = 1337,
                 **kwargs):
        super().__init__(
            model=model,
            data_dim=data_dim,
            missing_rate=missing_rate,
            mask_seed=mask_seed,
            **kwargs)


@teras_export("teras.models.TabTransformerRTDPretrainer")
class TabTransformerRTDPretrainer(backend.models.TabTransformerRTDPretrainer):
    """
    Replaced Token Detection (RTD) based Pretrainer for pretraining
    `TabTransformerBackbone` as proposed by Huang et al. in the paper,
    "TabTransformer: Tabular Data Modeling Using Contextual Embeddings".

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        model: keras.Model, instance of `TabTransformerBackbone` to
            pretrain
        data_dim: int, dimensionality of the input dataset
        replace_rate: float, fraction of original features to replace.
            Must be in the range [0, 1).
            Defaults to 0.3 (or 30%)
        mask_seed: int, seed for generating mask.
            Defaults to 1337
        shuffle_seed: int, seed for shuffling inputs.
            Defaults to 1999
    """
    def __init__(self,
                 model: keras.Model,
                 data_dim: int,
                 replace_rate: float = 0.3,
                 mask_seed: int = 1337,
                 shuffle_seed: int = 1999,
                 **kwargs):
        super().__init__(
            model=model,
            data_dim=data_dim,
            replace_rate=replace_rate,
            mask_seed=mask_seed,
            shuffle_seed=shuffle_seed,
            **kwargs)
