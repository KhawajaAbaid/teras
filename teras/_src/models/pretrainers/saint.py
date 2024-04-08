import keras

from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.models.SAINTPretrainer")
class SAINTPretrainer(backend.models.SAINTPretrainer):
    """
    SAINTPretrainer as proposed in the paper,
    "SAINT: Improved Neural Networks for Tabular Data".

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        model: keras.Model, instance of `SAINTBackbone` model to pretrain
        cardinalities: list, a list cardinalities of all the features
            in the dataset in the same order as the features' occurrence.
            For numerical features, use the value `0` as indicator at
            the corresponding index.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
        embedding_dim: int, dimensionality of the embeddings being used
             in the model.
        cutmix_probability: float, used by the `CutMix` layer in
            generation of mask that is used to mix samples together.
            Defaults to 0.3
        mixup_alpha: float, used by the `MixUp` layer in sampling from the
            `Beta` distribution which is then used to interpolate samples.
            Defaults to 1.
        temperature: float, used in the computation of the
            `contrastive_loss` to scale logits. Defaults to 0.7
        lambda_: float, acts as a weight when adding the contrastive loss
            and the denoising loss together.
            `loss = constrastive_loss + lambda_ * denoising_loss`
            Defaults to 10.
        lambda_c: float, used in the computation of the contrastive
            loss. Similar to `lambda_` is helps combined two sub-losses
            within the contrastive loss. Defaults to 0.5
        seed: int, seed used in random sampling and shuffling etc.
            It helps make the model behavior more deterministic.
            Defaults to (you guessed it) 1337.
    """
    def __init__(self,
                 model: keras.Model,
                 cardinalities: list,
                 embedding_dim: int,
                 cutmix_probability: float = 0.3,
                 mixup_alpha: float = 1.,
                 temperature: float = 0.7,
                 lambda_: float = 10.,
                 lambda_c: float = 0.5,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(model=model,
                         cardinalities=cardinalities,
                         embedding_dim=embedding_dim,
                         cutmix_probability=cutmix_probability,
                         mixup_alpha=mixup_alpha,
                         temperature=temperature,
                         lambda_=lambda_,
                         lambda_c=lambda_c,
                         seed=seed,
                         **kwargs)
