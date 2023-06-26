import tensorflow as tf
from tensorflow import keras


def info_nce_loss(real_projection_outputs=None,
                  augmented_projection_outputs=None,
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
    labels = tf.one_hot(tf.range(tf.shape(real_projection_outputs)[0]))
    logits_ab = tf.matmul(real_projection_outputs, augmented_projection_outputs, transpose_b=True) / temperature
    logits_ba = tf.matmul(augmented_projection_outputs, real_projection_outputs, transpose_b=True) / temperature
    loss_a = tf.losses.categorical_crossentropy(y_true=labels,
                                                y_pred=logits_ab,
                                                from_logits=True)
    loss_b = tf.losses.categorical_crossentropy(y_true=labels,
                                                y_pred=logits_ba,
                                                from_logits=True)
    loss = lambda_ * (loss_a + loss_b) / 2
    return loss


def denoising_loss(real_samples=None,
                   reconstructed_samples=None,
                   num_categorical_features: int = None):
    """
    Since we apply categorical and numerical embedding layers
    separately and then combine them into a new features matrix
    this effectively makes the first k features in the outputs
    categorical (since categorical embeddings are applied first)
    and all other features numerical.
    Here, k = num_categorical_features

    Args:
        real_samples: `tf.Tensor`,
            Samples drawn from the original dataset.
        reconstructed_samples: `tf.Tensor`,
            Samples reconstructed by the reconstruction head.
        num_categorical_features: `int`,
            Number of categorical features in the dataset.
            If there are no categorical features, specify 0.
    """
    if real_samples is None:
        raise ValueError("`real_samples` cannot be None. "
                         "You must pass the samples drawn from the original dataset. ")

    if reconstructed_samples is None:
        raise ValueError("`reconstructed_samples` cannot be None. "
                         "You must pass the samples reconstructed by the ReconstructionHead.")

    if num_categorical_features is None:
        raise ValueError("`num_categorical_features` cannot be None. "
                         "If there are no categorical features in the dataset, pass 0.")

    num_features = tf.shape(real_samples)[1]
    loss = 0.
    if num_categorical_features > 0:
        loss += tf.reduce_sum(tf.losses.categorical_crossentropy(real_samples,
                                                                 reconstructed_samples,
                                                                 from_logits=True))
    if num_categorical_features < num_features:
        # there are numerical features
        loss += tf.reduce_sum(tf.losses.mse(real_samples, reconstructed_samples))

    return loss
