import keras
from keras import random, ops
from keras.backend import floatx


class BaseGAIN(keras.Model):
    """
    Base class for GAIN.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 hint_rate: float = 0.9,
                 alpha: float = 100.,
                 **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.hint_rate = hint_rate
        self.alpha = alpha

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(
            name="discriminator_loss")

    def build(self, input_shape):
        # Inputs received by each generator and discriminator have twice the
        # dimensions of original inputs
        input_shape = (input_shape[:-1], input_shape[-1] * 2)
        self.generator.build(input_shape)
        self.discriminator.build(input_shape)

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(),
                discriminator_optimizer=keras.optimizers.Adam()
                ):
        super().compile()
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
        z = random.uniform(ops.shape(data), minval=0., maxval=0.01)
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
        })
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.layers.deserialize(config.pop("generator"))
        discriminator = keras.layers.deserialize(config.pop("discriminator"))
        return cls(generator=generator, discriminator=discriminator,
                   **config)
