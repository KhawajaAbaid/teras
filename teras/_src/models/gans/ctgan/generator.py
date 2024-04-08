from teras._src import backend
from teras._src.typing import IntegerSequence
from teras._src.api_export import teras_export


@teras_export("teras.models.CTGANGenerator")
class CTGANGenerator(backend.models.CTGANGenerator):
    """
    CTGANGenerator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: int, The dimensionality of the dataset.
            It will also be the dimensionality of the output produced
            by the generator.
            Note the dimensionality must be equal to the dimensionality of
            dataset that is passed to the fit method and not necessarily
            the dimensionality of the raw input dataset as sometimes
            data transformation alters the dimensionality of the dataset.
        metadata: dict, `CTGANGenerator` applies different activation functions
            to its outputs depending on the type of features (categorical or
            continuous). And to determine the feature types and for other
            computations during the activation step, the ``metadata``
            computed during the data transformation step, is required.
            It can be accessed through the `.metadata` property attribute of
            the `CTGANDataTransformer` instance which was used to transform
            the raw input data.
            Note that, this is NOT the same metadata as `features_metadata`,
            which is computed using the `get_metadata_for_embedding` utility
            function from `teras.utils`.
            You must access it through the `.metadata` property attribute of the
            `CTGANDataTransformer`.
        hidden_dims: Sequence, A sequence of integers that is used to
            construct the hidden block.
            For each value, a `CTGANGeneratorLayer` of that dimensionality is
            added. Defaults to [256, 256]
    """
    def __init__(self,
                 data_dim: int,
                 metadata: dict,
                 hidden_dims: IntegerSequence = (256, 256),
                 seed: int = 1337,
                 **kwargs):
        super().__init__(data_dim=data_dim,
                         metadata=metadata,
                         hidden_dims=hidden_dims,
                         seed=seed,
                         **kwargs)
