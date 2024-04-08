from tensorflow import keras

from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.models.TVAE")
class TVAE(backend.models.TVAE):
    """
    TVAE is a tabular data generation architecture proposed by Lei Xu et al.
    in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        encoder: keras.Model, an instance of `TVAEEncoder`. It encodes or
            compresses the input.
        decoder: keras.Model, an instance of `TVAEDecoder`. It decodes or
            decompresses from latent dimensions to data dimensions.
        latent_dim: int, Dimensionality of the learned latent space.
            Default 128.
        loss_factor: float, Hyperparameter used in the computation of
            `ELBO loss`. It controls how much the cross entropy loss
            contributes to the overall loss. It is directly proportional to
            the cross entropy loss. Defaults to 2.
    """
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 metadata: dict,
                 data_dim: int,
                 latent_dim: int = 128,
                 loss_factor: float = 2.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         metadata=metadata,
                         data_dim=data_dim,
                         latent_dim=latent_dim,
                         loss_factor=loss_factor,
                         seed=seed,
                         **kwargs)
