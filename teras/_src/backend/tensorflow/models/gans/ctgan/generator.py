from teras._src.backend.common.models.gans.ctgan.generator import BaseCTGANGenerator
from teras._src.typing import IntegerSequence


class CTGANGenerator(BaseCTGANGenerator):
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
