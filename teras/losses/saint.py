import tensorflow as tf
from tensorflow import keras


def info_nce_loss(projection_outputs_original=None,
                  projection_outputs_augmented=None,
                  temperature: float = 0.7,
                  lambda_: float = 0.5):
    """
    Contrastive Info NCE loss.
    In SAINT architecture, it is used in combination with the `denoising` loss.
    Args:
        projection_outputs_original: Outputs of projection head over encodings of original inputs
        projection_outputs_augmented: Outputs of projection head over the encodings of augmented inputs
        temperature: `float`, default 0.7,
            Temperature value used in the computation of the InfoNCE contrastive loss.
        lambda_: `float`, default 0.5,
           Used in combining the two losses.

    Returns:
        Info NCE loss.
    """
    labels = tf.one_hot(tf.range(tf.shape(projection_outputs_original)[0]))
    logits_ab = tf.matmul(projection_outputs_original, projection_outputs_augmented, transpose_b=True) / temperature
    logits_ba = tf.matmul(projection_outputs_augmented, projection_outputs_original, transpose_b=True) / temperature
    loss_a = tf.losses.categorical_crossentropy(y_true=labels,
                                                y_pred=logits_ab,
                                                from_logits=True)
    loss_b = tf.losses.categorical_crossentropy(y_true=labels,
                                                y_pred=logits_ba,
                                                from_logits=True)
    loss = lambda_ * (loss_a + loss_b) / 2
    return loss


