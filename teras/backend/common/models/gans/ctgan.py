import keras
from keras import random
from teras.losses.ctgan import (ctgan_discriminator_loss,
                                ctgan_generator_loss)
from teras.utils.utils import clean_reloaded_config_data


class BaseCTGAN(keras.Model):
    """
    Base CTGAN model class.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 metadata: dict,
                 latent_dim: int = 128,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.metadata = metadata
        self.latent_dim = latent_dim
        self.seed = seed
        self._seed_gen = random.SeedGenerator(self.seed)

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(
            name="discriminator_loss")

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(
                    learning_rate=1e-3,
                    beta_1=0.5, beta_2=0.9,
                    name="generator_optimizer"
                ),
                discriminator_optimizer=keras.optimizers.Adam(
                    learning_rate=1e-3,
                    beta_1=0.5, beta_2=0.9,
                    name="discriminator_optimizer"
                ),
                generator_loss=ctgan_generator_loss,
                discriminator_loss=ctgan_discriminator_loss,
                **kwargs
                ):
        super().compile(**kwargs)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def build(self, input_shape):
        self.discriminator.build(input_shape)
        # Generator receives the input of dimensons = latent_dim + |cond_vector|
        # where, |cond_vector| = total_num_categories
        batch_size, input_dim = input_shape
        input_shape = (batch_size,
                       self.latent_dim +
                       self.metadata["categorical"]["total_num_categories"])
        self.generator.build(input_shape)

    @property
    def metrics(self):
        return [self.generator_loss_tracker,
                self.discriminator_loss_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({
            'generator': keras.layers.serialize(self.generator),
            'discriminator': keras.layers.serialize(self.generator),
            'latent_dim': self.latent_dim,
            'seed': self.seed,
        }
        )
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.layers.deserialize(config.pop("generator"))
        discriminator = keras.layers.deserialize(config.pop("discriminator"))
        metadata = clean_reloaded_config_data(config.pop("metadata"))
        return cls(generator=generator,
                   discriminator=discriminator,
                   metadata=metadata,
                   **config)
