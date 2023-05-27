import tensorflow as tf


def generator_loss(y_generated, cond_vector, mask=None, features_meta_data=None):
    loss = []
    continuous_features_relative_indices = features_meta_data["continuous"]["relative_indices_all"]
    features_relative_indices_all = features_meta_data["relative_indices_all"]
    num_categories_all = features_meta_data["categorical"]["num_categories_all"]
    # the first k features in the data are continuous which we'll ignore as we're only
    # concerned with the categorical features here
    offset = len(continuous_features_relative_indices)
    for i, index in enumerate(features_relative_indices_all[offset:]):
        logits = y_generated[:, index: index + num_categories_all[i]]
        temp_cond_vector = cond_vector[:, i: i + num_categories_all[i]]
        labels = tf.expand_dims(tf.argmax(temp_cond_vector, axis=1), 1)
        crossentropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=labels
                                                                    )
        loss.append(crossentropy_loss)
    loss = tf.stack(loss, axis=1)
    loss = tf.reduce_sum(loss * tf.cast(mask, dtype=tf.float32)) / tf.cast(tf.shape(y_generated)[0], dtype=tf.float32)
    loss = -tf.reduce_mean(y_generated) * loss
    return loss


def discriminator_loss(y_real, y_generated):
    """
    CTGAN's discriminator loss as proposed by xyz et al.
    in abc paper.
    Args:
        y_real: Discriminator's output for real samples
        y_generated: Discriminator's output for generated samples
    """
    return -(tf.reduce_mean(y_real) - tf.reduce_mean(y_generated))


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