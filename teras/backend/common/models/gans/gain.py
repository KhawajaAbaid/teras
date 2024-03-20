import keras
from keras import random, ops


class BaseGAIN(keras.Model):
    """
    Base class for GAIN.

    # TODO remove the rest of the docstring and move to the wrapper class
    GAIN is a missing data imputation model based on GANs. This is an
    implementation of the GAIN architecture proposed by Jinsung Yoon et al.
    in the paper,
    "GAIN: Missing Data Imputation using Generative Adversarial Nets"

    In GAIN, the generator observes some components of a real data vector,
    imputes the missing components conditioned on what is actually observed, and
    outputs a completed vector.
    The discriminator then takes a completed vector and attempts to determine
    which components were actually observed and which were imputed. It also
    utilizes a novel hint mechanism, which ensures that generator does in
    fact learn to generate samples according to the true data distribution.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        generator: keras.Model, An instance of `GAINGenerator` model or any
            customized model that can work in its place.
        discriminator: keras.Model, An instance of `GAINDiscriminator` model
            or any customized model that can work in its place.
        hint_rate: float, Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Must be between 0. and 1.
            Hint vectors ensure that generated samples follow the underlying
            data distribution.
            Defaults to 0.9

        alpha: float, Hyper parameter for the generator loss computation that
            controls how much weight should be given to the MSE loss.
            Precisely,
            `generator_loss` = `cross_entropy_loss` + `alpha` * `mse_loss`
            The higher the `alpha`, the more the mse_loss will affect the
            overall generator loss.
            Defaults to 100.
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
        self.loss_tracker = keras.metrics.Mean(
            name="loss")
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
        metrics = [self.loss_tracker,
                   self.generator_loss_tracker,
                   self.discriminator_loss_tracker,
                   ]
        return metrics

    def compute_loss(self):
        # TODO
        raise NotImplementedError

    def compute_generator_loss(self, x, x_generated, mask, mask_pred, alpha):
        cross_entropy_loss = keras.losses.CategoricalCrossentropy()(
            mask, mask_pred
        )
        mse_loss = keras.losses.MeanSquaredError()(
            y_true=(mask * x),
            y_pred=(mask * x_generated))
        loss = cross_entropy_loss + alpha * mse_loss
        return loss

    def call(self):
        # TODO
        raise NotImplementedError

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
        # imputed_data = self.generator(tf.concat([x, mask], axis=1))
        imputed_data = self(x, mask=mask)
        imputed_data = mask * data + (1 - mask) * imputed_data
        return imputed_data

    def impute(self,
               x,
               data_transformer=None,
               reverse_transform=True,
               batch_size=None,
               verbose="auto",
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False
               ):
        """
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
        new_config = {'generator': keras.layers.serialize(self.generator),
                      'discriminator': keras.layers.serialize(self.discriminator)
                      }
        config.update(new_config)
        return config
