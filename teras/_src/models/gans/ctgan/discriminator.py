from teras._src import backend
from teras._src.typing import IntegerSequence
from teras._src.api_export import teras_export


@teras_export("teras.models.CTGANDiscriminator")
class CTGANDiscriminator(backend.models.CTGANDiscriminator):
    """
    CTGANDiscriminator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        hidden_dims: Sequence, A sequence of integers that is used to
            construct the hidden block.
            For each value, a `CTGANDiscriminatorLayer` of that
            dimensionality is added. Defaults to [256, 256]
        packing_degree: int, Packing degree - taken from the PacGAN paper.
            The number of samples concatenated or "packed" together.
            It must be a factor of the batch_size.
            Packing degree is borrowed from the PacGAN
            [Diederik P. Kingma et al.] architecture,
            which proposes passing `m` samples at once to discriminator
            instead of `1` to be
            jointly classified as real or fake by the discriminator in
            order to tackle the
            issue of mode collapse inherent in the GAN based architectures.
            The number of samples passed jointly `m`, is termed as the
            `packing degree`.
            Defaults to 8.
        gradient_penalty_lambda: float, Controls the strength of gradient
            penalty. lambda value is directly proportional to the strength of
            gradient penalty.
            Gradient penalty penalizes the discriminator for large
            weights in an attempt to combat Discriminator becoming
            too confident and overfitting.
            Defaults to 10.
        seed: int, Seed for random sampling. Defaults to 1337.
    """
    def __init__(self,
                 hidden_dims: IntegerSequence = (256, 256),
                 packing_degree: int = 8,
                 gradient_penalty_lambda: float = 10.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(hidden_dims=hidden_dims,
                         packing_degree=packing_degree,
                         gradient_penalty_lambda=gradient_penalty_lambda,
                         seed=seed,
                         **kwargs)