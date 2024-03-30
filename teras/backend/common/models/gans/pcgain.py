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
                 hint_rate: float = 0.9,
                 alpha: float = 100.,
                 **kwargs):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         **kwargs)

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
