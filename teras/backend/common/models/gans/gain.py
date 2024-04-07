import keras
from keras import random, ops
from keras.backend import floatx
from keras.backend import backend


class BaseGAIN(keras.Model):
    """
    Base class for GAIN.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 hint_rate: float = 0.9,
                 alpha: float = 100.,
                 seed: int = 1337,
                 **kwargs):
        if not backend() == "jax":
             # Don't call super() with JAX backend as `JAXGAN` does that
             # already!
             super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.seed = seed

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(
            name="discriminator_loss")

        self._seed_gen = random.SeedGenerator(self.seed)

    def build(self, input_shape):
        # Inputs received by each generator and discriminator have twice the
        # dimensions of original inputs
        input_shape = (input_shape[:-1], input_shape[-1] * 2)
        self.generator.build(input_shape)
        self.discriminator.build(input_shape)

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(),
                discriminator_optimizer=keras.optimizers.Adam(),
                **kwargs
                ):
        super().compile(**kwargs)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    @property
    def metrics(self):
        metrics = [
                   self.generator_loss_tracker,
                   self.discriminator_loss_tracker,
                   ]
        return metrics

    def compute_loss(self, **kwargs):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` doesn't provide an implementation for"
            f" the `compute_loss` method. Please use "
            f"`compute_discriminator_loss` or `compute_generator_loss` for "
            f"relevant purpose."
        )

    def compute_discriminator_loss(self, mask, mask_pred):
        return keras.losses.BinaryCrossentropy()(mask, mask_pred)

    def compute_generator_loss(self, x, x_generated, mask, mask_pred, alpha):
        cross_entropy_loss = keras.losses.CategoricalCrossentropy()(
            mask, mask_pred
        )
        mse_loss = keras.losses.MeanSquaredError()(
            y_true=(mask * x),
            y_pred=(mask * x_generated))
        loss = cross_entropy_loss + alpha * mse_loss
        return loss

    def call(self, **kwargs):
        raise NotImplementedError(
            f"`{self.__class__.__name__}` doesn't provide an implementation "
            f"for the `call` method. Please use the call method of "
            f"`GAIN().generator` or `GAIN().discriminator`."
        )

    def test_step(self, data):
        x_gen, x_disc = data

        # =========================
        # Test the discriminator
        # =========================
        # Create mask
        mask = 1. - ops.cast(ops.isnan(x_disc), dtype=floatx())
        # replace nans with 0.
        x_disc = ops.where(ops.isnan(x_disc), x1=0., x2=x_disc)
        # Sample noise
        z = random.uniform(shape=ops.shape(x_disc), minval=0.,
                           maxval=0.01, seed=self._seed_gen)
        # Sample hint vectors
        hint_vectors = random.binomial(shape=ops.shape(x_disc),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=self._seed_gen)
        hint_vectors *= mask
        # Combine random vectors with original data
        x_disc = x_disc * mask + (1 - mask) * z
        x_generated = self.generator(ops.concatenate([x_disc, mask], axis=1))
        # Combine generated samples with original data
        x_hat_disc = (x_generated * (1 - mask)) + (x_disc * mask)
        mask_pred = self.discriminator(
            ops.concatenate([x_hat_disc, hint_vectors], axis=1))
        d_loss = self.compute_discriminator_loss(mask, mask_pred)

        # =====================
        # Test the generator
        # =====================
        mask = 1. - ops.cast(ops.isnan(x_gen), dtype=floatx())
        x_gen = ops.where(ops.isnan(x_gen), x1=0., x2=x_gen)
        z = random.uniform(shape=ops.shape(x_gen), minval=0.,
                           maxval=0.01, seed=self._seed_gen)
        hint_vectors = random.binomial(shape=ops.shape(x_gen),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=self._seed_gen)
        hint_vectors *= mask
        x_gen = x_gen * mask + (1 - mask) * z

        x_generated = self.generator(
            ops.concatenate([x_gen, mask], axis=1))
        # Combine generated samples with original/observed data
        x_hat = (x_generated * (1 - mask)) + (x_gen * mask)
        mask_pred = self.discriminator(
            ops.concatenate([x_hat, hint_vectors], axis=1))
        g_loss = self.compute_generator_loss(
            x=x_gen,
            x_generated=x_generated,
            mask=mask,
            mask_pred=mask_pred,
            alpha=self.alpha
        )

        # Update custom tracking metrics
        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)

        logs = {m.name: m.result() for m in self.metrics}
        return logs

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def predict_step(self, data):
        """
        Args:
            Transformed data.
        Returns:
            Imputed data that should be reverse transformed
            to its original form.
        """
        if isinstance(data, tuple):
            data = data[0]
        data = ops.cast(data, floatx())
        # Create mask
        mask = 1. - ops.cast(ops.isnan(data), dtype=floatx())
        data = ops.where(ops.isnan(data), x1=0., x2=data)
        # Sample noise
        z = random.uniform(ops.shape(data), minval=0.,
                           maxval=0.01, seed=self._seed_gen)
        x = mask * data + (1 - mask) * z
        imputed_data = self.generator(ops.concatenate([x, mask], axis=1))
        imputed_data = mask * data + (1 - mask) * imputed_data
        return imputed_data

    def get_config(self):
        config = super().get_config()
        config.update({
            'generator': keras.layers.serialize(self.generator),
            'discriminator': keras.layers.serialize(self.discriminator),
            'hint_rate': self.hint_rate,
            'alpha': self.alpha,
            'seed': self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.layers.deserialize(config.pop("generator"))
        discriminator = keras.layers.deserialize(config.pop("discriminator"))
        return cls(generator=generator, discriminator=discriminator,
                   **config)
