import keras
from keras import random, ops
from teras._src.backend.common.models.gans.gain import BaseGAIN
from keras.backend import floatx


class BasePCGAIN(BaseGAIN):
    """
    Base class for PCGAIN.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 classifier: keras.Model,
                 hint_rate: float = 0.9,
                 alpha: float = 200.,
                 beta: float = 100.,
                 seed: int = 1337,
                 **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         seed=seed,
                         **kwargs)
        self.classifier = classifier
        self.beta = beta

    def build(self, input_shape):
        super().build(input_shape)
        # we want to build classifier last so its variables are at the end of
        # lists for both trainable and non-trainable variables lists
        self.classifier.build(input_shape)

    def compute_generator_loss(self, x, x_generated, mask, mask_pred,
                               classifier_pred, alpha, beta):
        cross_entropy_loss = keras.losses.BinaryCrossentropy()(
            mask, mask_pred
        )
        mse_loss = keras.losses.MeanSquaredError()(
            y_true=(mask * x),
            y_pred=(mask * x_generated))

        info_entropy_loss = -ops.mean(
            classifier_pred * ops.log(classifier_pred + 1e-8))
        loss = cross_entropy_loss + (alpha * mse_loss) + (beta *
                                                          info_entropy_loss)
        return loss

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
        z = random.uniform(shape=ops.shape(x_disc), minval=0., maxval=0.01,
                           seed=self._seed_gen)
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
        z = random.uniform(shape=ops.shape(x_gen), minval=0., maxval=0.01,
                           seed=self._seed_gen)
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
        classifier_pred = self.classifier(x_hat)
        mask_pred = self.discriminator(
            ops.concatenate([x_hat, hint_vectors], axis=1))
        g_loss = self.compute_generator_loss(
            x=x_gen,
            x_generated=x_generated,
            mask=mask,
            mask_pred=mask_pred,
            classifier_pred=classifier_pred,
            alpha=self.alpha,
            beta=self.beta,
        )

        # Update custom tracking metrics
        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)

        logs = {m.name: m.result() for m in self.metrics}
        return logs

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
        z = random.uniform(ops.shape(data), minval=0., maxval=0.01,
                           seed=self._seed_gen)
        x = mask * data + (1 - mask) * z
        imputed_data = self.generator(ops.concatenate([x, mask], axis=1))
        imputed_data = mask * data + (1 - mask) * imputed_data
        return imputed_data

    def get_config(self):
        config = super().get_config()
        config.update({
            'classifier': keras.layers.serialize(self.classifier),
            'beta': keras.layers.serialize(self.beta),
        })
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.layers.deserialize(config.pop("generator"))
        discriminator = keras.layers.deserialize(config.pop("discriminator"))
        classifier = keras.layers.deserialize(config.pop("classifier"))
        return cls(generator=generator, discriminator=discriminator,
                   classifier=classifier, **config)
