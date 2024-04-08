import keras

from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.models.GAIN")
class GAIN(backend.models.GAIN):
    """
    GAIN is a missing data imputation model based on GANs. This is an
    implementation of the GAIN architecture proposed by Jinsung Yoon et al.
    in the paper,
    "GAIN: Missing Data Imputation using Generative Adversarial Nets"

    In GAIN, the generator observes some components of a real data vector,
    imputes the missing components conditioned on what is actually observed, and
    outputs a completed vector.
    The discriminator then takes a completed vector and attempts to determine
    which components were actually observed and which were imputed. It also
    utilizes a novel hint mechanism, which ensures that generator does in
    fact learn to generate samples according to the true data distribution.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        generator: keras.Model, An instance of `GAINGenerator` model or any
            customized model that can work in its place.
        discriminator: keras.Model, An instance of `GAINDiscriminator` model
            or any customized model that can work in its place.
        hint_rate: float, Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Must be between 0. and 1.
            Hint vectors ensure that generated samples follow the underlying
            data distribution.
            Defaults to 0.9
        alpha: float, Hyper parameter for the generator loss computation that
            controls how much weight should be given to the MSE loss.
            Precisely,
            `generator_loss` = `cross_entropy_loss` + `alpha` * `mse_loss`
            The higher the `alpha`, the more the mse_loss will affect the
            overall generator loss.
            Defaults to 100.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 hint_rate: float = 0.9,
                 alpha: float = 100.,
                 **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         **kwargs)
