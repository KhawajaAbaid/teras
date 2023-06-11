import tensorflow as tf
from tensorflow import keras


@tf.function
def generator_loss(generated_samples, y_generated,
                   cond_vector=None, mask=None, features_meta_data=None):
    loss = []
    cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction=keras.losses.Reduction.NONE)
    continuous_features_relative_indices = features_meta_data["continuous"]["relative_indices_all"]
    features_relative_indices_all = features_meta_data["relative_indices_all"]
    num_categories_all = features_meta_data["categorical"]["num_categories_all"]
    # the first k features in the data are continuous which we'll ignore as we're only
    # concerned with the categorical features here
    offset = len(continuous_features_relative_indices)
    for i, index in enumerate(features_relative_indices_all[offset:]):
        logits = generated_samples[:, index: index + num_categories_all[i]]
        temp_cond_vector = cond_vectors[:, i: i + num_categories_all[i]]
        labels = tf.argmax(temp_cond_vector, axis=1)
        ce_loss = cross_entropy_loss(y_pred=logits,
                                     y_true=labels
                                     )
        loss.append(ce_loss)
    loss = tf.stack(loss, axis=1)
    loss = tf.reduce_sum(loss * tf.cast(mask, dtype=tf.float32)) / tf.cast(tf.shape(y_generated)[0], dtype=tf.float32)
    loss = -tf.reduce_mean(y_generated) * loss
    return loss


@tf.function
def discriminator_loss(y_real, y_generated):
    """
    Loss for the Discriminator model in the CTGAN architecture.

    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        y_real: Discriminator's output for real samples
        y_generated: Discriminator's output for generated samples

    Returns:
        Discriminator's loss.
    """
    return -(tf.reduce_mean(y_real) - tf.reduce_mean(y_generated))


@tf.function
def generator_dummy_loss(y_dummy, y_pred):
    """
    For the generator model to track the loss function, and show it in outputs
    we create a dummy loss function which receives the loss function
    and returns it as is. It is passed to the model during compilation step.

    Reference(s):
        Idea taken from:
        https://towardsdatascience.com/solving-the-tensorflow-keras-model-loss-problem-fd8281aeeb11

    Args:
        y_dummy: An array of length batch_size, filled with dummy values.
        y_pred: The loss value computed using a custom loss function.

    Returns:
        Returns y_pred (i.e. loss) as is.
    """
    return tf.squeeze(y_pred)