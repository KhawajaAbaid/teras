import tensorflow as tf


def reconstruction_loss(real_samples=None,
                        reconstructed_samples=None,
                        mask=None):
    """
    Reconstruction loss for TabNet Pretrainer mode as propsoed by
     Sercan et al. in the paper,
     "TabNet: Attentive Interpretable Tabular Learning"

     Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        real_samples: Samples drawn from the input dataset
        reconstructed_samples: Samples reconstructed by the decoder
        mask: Mask that indicates the missing of features in a sample

    Returns:
        Reconstruction loss for TabNet Pretraining.
    """
    nominator_part = (reconstructed_samples - real_samples) * mask
    real_samples_population_std = tf.math.reduce_std(tf.cast(real_samples, dtype=tf.float32))
    # divide
    x = nominator_part / real_samples_population_std
    # Calculate L2 norm
    loss = tf.sqrt(tf.reduce_sum(tf.square(x)))
    return loss