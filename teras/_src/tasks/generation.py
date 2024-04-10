import keras
from keras import random, ops
from teras._src.models.gans.ctgan.ctgan import CTGAN
from teras._src.models.gans.ctgan.generator import CTGANGenerator
from teras._src.api_export import teras_export


@teras_export("teras.tasks.Generator")
class Generator:
    """
    Generator class that provides methods related to data generation.

    Args:
        model: keras.Model, instance of the trained model that will be
            used to generate data
        data_transformer: Instance of data transformer used to transform data
            for training.
        data_sampler: Instance of the data sampler used to sample data for
            training.
    """
    def __init__(self,
                 model: keras.Model,
                 data_transformer,
                 data_sampler=None,
                 ):
        self.model = model
        self.data_transformer = data_transformer
        self.data_sampler = data_sampler

    def generate(self, num_samples, latent_dim, batch_size=None,
                 verbose="auto", steps=None, callbacks=None, seed=None):
        """
        Generates new data samples.
        It exposes all the arguments taken by the `predict` method.

        Args:
            num_samples: int, number of samples to generate.
            latent_dim: int, latent dimensions for sampling noise. It should
                be the same as used in the model during training.
        """
        z = random.normal((num_samples, latent_dim), seed=seed)
        if isinstance(self.model, (CTGAN, CTGANGenerator)):
            if self.data_sampler is None:
                raise ValueError(
                    "For `CTGAN` architecture `data_sampler` cannot be `None`."
                    "you must pass the data sampler instance that was used to "
                    "train the architecture. "
                    f"Received: {self.data_sampler}"
                )
            cond_vectors = self.data_sampler.sample_cond_vectors_for_generation(
                batch_size=num_samples
            )
            z = ops.concatenate([z, cond_vectors], axis=1)

        x_generated = self.model.predict(z,
                                         batch_size=batch_size,
                                         verbose=verbose,
                                         steps=steps,
                                         callbacks=callbacks
                                         )
        x_generated = self.data_transformer.reverse_transform(x_generated)
        return x_generated

