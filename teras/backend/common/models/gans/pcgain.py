import keras
from keras import random, ops
from teras.backend.common.models.gans.gain import BaseGAIN


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
                 **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         **kwargs)
        self.classifier = classifier
        self.beta = beta

    def build(self, input_shape):
        # Inputs received by each generator and discriminator have twice the
        # dimensions of original inputs
        input_shape = (input_shape[:-1], input_shape[-1] * 2)
        self.generator.build(input_shape)
        self.discriminator.build(input_shape)
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
