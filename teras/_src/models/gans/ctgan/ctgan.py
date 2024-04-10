import keras

from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.models.CTGAN")
class CTGAN(backend.models.CTGAN):
    """
    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        generator: keras.Model, An instance of :py:class:`CTGANGenerator`.
        discriminator: keras.Model, An instance of
            :py:class:`CTGANDiscriminator`.
        metadata: dict, A dictionary containing features metadata computed
            during the data transformation step.
            It can be accessed through the `.metadata` property attribute of
            the :py:class:`CTGANDataTransformer` instance which was used to
            transform the raw input data.
            Note that, this is NOT the same metadata as `features_metadata`,
            which is computed using the `get_metadata_for_embedding` utility
            function from :py:mod:`teras.utils`.
        latent_dim: int, Dimensionality of noise or `z` that serves as
            input to :py:class:`CTGANGenerator` to generate samples.
            Defaults to 128.
        seed: int, Seed for random sampling. Defaults to 1337.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 metadata: dict,
                 latent_dim: int = 128,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         metadata=metadata,
                         latent_dim=latent_dim,
                         seed=seed,
                         **kwargs)
