import tensorflow as tf
from tensorflow.keras import losses
from teras.losses.gain import (generator_loss as generator_pretraining_loss,
                               discriminator_loss)


# The generator pretraining loss is the same as generator_loss
# in the GAIN model because in PC-GAIN we use a GAIN model for
# pretraining.

# In PC-GAIN, the discriminator loss is also exactly the same
# as GAIN's discriminator loss

def generator_loss(real_samples=None,
                   generated_samples=None,
                   discriminator_pred=None,
                   mask=None,
                   alpha=200,
                   beta=100,
                   classifier_pred=None):
    """
    Generator loss used during the main training phase (post-pretraining)
    of PC-GAIN architecture.
    It is similar to the generator loss used in the pretraining stage
    except for an additional `Information Entropy Loss` that is calculated
    for the Classifier's predictions and weighted by the `beta` parameter.

    Args:
        real_samples: yep
        generated_samples: yep
        discriminator_pred: Predictions by discriminator
        mask: yep
        alpha: yep
        beta: yep
        classifier_pred: Softmax probs predicted by classifier
    """
    cross_entropy_loss = -tf.reduce_mean((1 - mask) * tf.math.log(discriminator_pred + 1e-8))
    mse_loss = losses.MSE(y_true=(mask * real_samples),
                                  y_pred=(mask * generated_samples))
    info_entropy_loss = -tf.reduce_mean(classifier_pred * tf.math.log(classifier_pred + 1e-8))
    loss = cross_entropy_loss + (alpha * mse_loss) + (beta * info_entropy_loss)
    return loss
