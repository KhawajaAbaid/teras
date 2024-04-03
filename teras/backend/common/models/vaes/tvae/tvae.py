import keras
from keras import random, ops


class BaseTVAE(keras.Model):
    """
    Base TVAE class.
    """
    def __init__(self,
                 encoder: keras.Model,
                 decoder: keras.Model,
                 latent_dim: int = 128,
                 loss_factor: float = 2.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.loss_factor = loss_factor
        self.seed = seed
        self._seed_gen = random.SeedGenerator(self.seed)

    def call(self, inputs):
        mean, log_var, std = self.encoder(inputs)
        eps = random.uniform(shape=ops.shape(std),
                             minval=0, maxval=1,
                             dtype=std.dtype,
                             seed=self._seed_gen)
        z = (std * eps) + mean
        generated_samples, sigmas = self.decoder(z)
        return generated_samples, sigmas

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': keras.layers.serialize(self.encoder),
            'decoder': keras.layers.serialize(self.decoder),
            'latent_dim': self.latent_dim,
            'loss_factor': self.loss_factor,
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = keras.layers.deserialize(config.pop("encoder"))
        decoder = keras.layers.deserialize(config.pop("decoder"))
        return cls(encoder=encoder,
                   decoder=decoder,
                   **config)
