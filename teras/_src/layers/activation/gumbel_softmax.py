import keras
from teras._src.activations import gumbel_softmax
from teras._src.api_export import teras_export


@teras_export("teras.layers.GumbelSoftmax")
class GumbelSoftmax(keras.layers.Layer):
    """
    Implementation of the Gumbel Softmax activation
    proposed by Eric Jang et al. in the paper,
    "Categorical Reparameterization with Gumbel-Softmax"

    Reference(s):
        https://arxiv.org/abs/1611.01144

    Args:
        temperature: float, Controls the sharpness or smoothness of the
            resulting probability distribution. A higher temperature value
            leads to a smoother and more uniform probability distribution.
            Conversely, a lower temperature value makes the distribution
            concentrated around the category with the highest probability.
        hard: bool, Whether to return soft probabilities or hard one hot
        vectors. Defaults to False.
        seed: int, Seed for random sampling.
    """
    def __init__(self,
                 temperature: float = 0.2,
                 hard: bool = False,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard
        self.seed = seed

    def build(self, input_shape):
        # nothing to build
        self.built = True

    def call(self, logits):
        return gumbel_softmax(logits,
                              temperature=self.temperature,
                              hard=self.hard,
                              seed=self.seed)

    def get_config(self):
        config = super().get_config()
        new_config = {'temperature': self.temperature,
                      'hard': self.hard}
        config.update(new_config)
        return config
