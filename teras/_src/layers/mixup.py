import keras
from keras import random, ops
from teras._src.api_export import teras_export


@teras_export("teras.layers.MixUp")
class MixUp(keras.layers.Layer):
    """
    MixUp is a regularization layer proposed by Hongyi Zhang et al.
    in the paper,
    mixup: BEYOND EMPIRICAL RISK MINIMIZATION

    It was originally proposed for image data but here it has been
    adapted for Tabular data.

    Reference(s):
        https://arxiv.org/abs/1710.09412

    Args:
        alpha: float, Parameter for the Beta distribution to sample
            `lambda_` from which is used to interpolate samples.
        lambda_seed: int, seed for sampling `lambda_` value from beta
            distribution. Defaults to 1337
        shuffle_seed: int, seed for randomly shuffling inputs.
            Defaults to 1999
    """
    def __init__(self,
                 alpha: float = 1.,
                 lambda_seed: int = 1337,
                 shuffle_seed: int = 1999,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_seed = lambda_seed
        self._lambda_seed_gen = random.SeedGenerator(self.lambda_seed)
        self.shuffle_seed = shuffle_seed
        self._shuffle_seed_gen = random.SeedGenerator(self.shuffle_seed)

    def build(self, input_shape):
        # there's nothing to build lol
        pass

    def call(self, inputs):
        # Sample lambda_
        lambda_ = ops.squeeze(random.beta(shape=(1,),
                                          alpha=self.alpha,
                                          beta=self.alpha,
                                          seed=self._lambda_seed_gen))
        # For each data sample select a partner to mix it with at random.
        # To efficiently achieve this, we can just shuffle the data
        random_partners = random.shuffle(inputs,
                                         axis=0,
                                         seed=self._shuffle_seed_gen)

        inputs_mixedup = (lambda_ * inputs) + (1 - lambda_) * random_partners
        return inputs_mixedup

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "lambda_seed": self.lambda_seed,
            "shuffle_seed": self.shuffle_seed,
        })
        return config
