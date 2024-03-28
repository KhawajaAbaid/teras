import keras
from keras import random, ops


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
            "GAIN doesn't provide an implementation for the `compute_loss` "
            "method. Please use `compute_discriminator_loss` or "
            "`compute_generator_loss` for relevant purpose."
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
            "`GAIN` doesn't provide an implementation for the `call` method. "
            "Please use the call method of `GAIN().generator` or "
            "`GAIN().discriminator`."
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
        data = ops.cast(data, "float32")
        # Create mask
        mask = 1. - ops.cast(ops.isnan(data), dtype="float32")
        data = ops.where(ops.isnan(data), x1=0., x2=data)
        # Sample noise
        z = random.uniform(ops.shape(data), minval=0., maxval=0.01)
        x = mask * data + (1 - mask) * z
        imputed_data = self.generator(ops.concatenate([x, mask], axis=1))
        imputed_data = mask * data + (1 - mask) * imputed_data
        return imputed_data

    def impute(self, x, data_transformer=None, reverse_transform=True,
               batch_size=None, verbose="auto", steps=None, callbacks=None,
               max_queue_size=10, workers=1, use_multiprocessing=False):
        """
        # TODO move to a separate Imputer task class
        Impute function combines GAIN's `predict` method and
        DataTransformer's `reverse_transform` method to fill
        the missing data and transform into the original format.
        It exposes all the arguments taken by the `predict` method.

        Args:
            x: pd.DataFrame, dataset with missing values.
            data_transformer: An instance of DataTransformer class which
                was used in transforming the raw input data
            reverse_transform: bool, default True, whether to reverse
                transformed the raw imputed data to original format.

        Returns:
            Imputed data in the original format.
        """
        x_imputed = self.predict(x,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 steps=steps,
                                 callbacks=callbacks,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)
        if reverse_transform:
            if data_transformer is None:
                raise ValueError(
                    "To reverse transform the raw imputed data, "
                    "`data_transformer` must not be None. "
                    "Please pass the instance of DataTransformer class used to "
                    "transform the input data as argument to this `impute` "
                    "method. \n"
                    "Or alternatively, you can set `reverse_transform` "
                    "parameter to False, and manually reverse transform the "
                    "generated raw data to original format using the "
                    "`reverse_transform` method of `DataTransformer` instance.")
            x_imputed = data_transformer.reverse_transform(x_imputed)
        return x_imputed

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
