import keras
from keras import random, ops
from teras._src.api_export import teras_export
from keras import backend


@teras_export("teras.layers.CutMix")
class CutMix(keras.layers.Layer):
    """
    CutMix is a regularization layer proposed by Sangdoo Yun et al.
    in the paper,
    CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features

    It was originally proposed for image data but here it has been
    adapted for Tabular data.

    Args:
        probability: float, CutMix probability which is used in
            generation of mask that is used to mix samples together.
            Defaults to 0.3
        mask_seed: int, seed used in the generation fo the mask
            Defaults to 1337
        shuffle_seed: int, seed used in shuffling the inputs
            Defaults to 1999
    """
    def __init__(self,
                 probability: float = 0.3,
                 mask_seed: int = 1337,
                 shuffle_seed: int = 1999,
                 **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.mask_seed = mask_seed
        self._mask_seed_gen = random.SeedGenerator(self.mask_seed)
        self.shuffle_seed = shuffle_seed
        self._shuffle_seed_gen = random.SeedGenerator(self.shuffle_seed)

    def build(self, input_shape):
        # there's nothing to build lol
        pass

    def call(self, inputs):
        # Generate mask for CutMix mixing
        mask_cutmix = random.binomial(ops.shape(inputs),
                                      counts=1,
                                      probabilities=self.probability,
                                      seed=self._mask_seed_gen)

        # For each data sample select a partner to mix it with at random.
        # To efficiently achieve this, we can just shuffle the data
        random_partners = random.shuffle(inputs,
                                         axis=0,
                                         seed=self._shuffle_seed_gen)

        # Apply cutmix formula
        inputs_cutmixed = (inputs * mask_cutmix) + (random_partners * (1 - mask_cutmix))
        return inputs_cutmixed

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "probability": self.probability,
            "mask_seed": self.mask_seed,
            "shuffle_seed": self.shuffle_seed,
        })
        return config
