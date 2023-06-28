import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses


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
    batch_size = tf.shape(real_projection_outputs)[0]
    labels = tf.range(batch_size)
    logits_ab = tf.matmul(real_projection_outputs, augmented_projection_outputs, transpose_b=True) / temperature
    logits_ba = tf.matmul(augmented_projection_outputs, real_projection_outputs, transpose_b=True) / temperature
    loss_a = losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=labels,
                                                                    y_pred=logits_ab)
    loss_b = losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=labels,
                                                                    y_pred=logits_ba)
    loss = lambda_ * (loss_a + loss_b) / 2
    return loss


def denoising_loss(real_samples=None,
                   reconstructed_samples=None,
                   categorical_features_metadata: dict = None):
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

    if categorical_features_metadata is None:
        raise ValueError("`categorical_features_metadata` cannot be None. ")

    num_features = tf.shape(real_samples)[1]
    num_categorical_features = len(categorical_features_metadata)
    num_categories_per_feature = list(map(lambda x: len(x[1]), categorical_features_metadata.values()))
    total_categories = sum(num_categories_per_feature)
    loss = 0.
    # The reconstructed samples have dimensions equal to number of categorical features + number of categories in all
    # the categorical features combined.
    # In other words, each categorical feature gets expanded into `number of categories` features due to softmax layer
    # during the reconstruction phase.
    # Since the order they occur in is still the same
    # except for that the first `total_categories` number of features are categorical and following are numerical
    for current_idx, (feature_idx, _) in enumerate(categorical_features_metadata.values()):
        num_categories = num_categories_per_feature[current_idx]
        loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                                    y_true=real_samples[:, feature_idx],  # real_samples have the original feature order
                                    y_pred=reconstructed_samples[:, current_idx: current_idx + num_categories])
        current_idx += 1
    # Check if there are numerical features -- if so, compute the mse loss
    mse_loss = tf.cond(pred=num_categorical_features < num_features,
                       true_fn=lambda: tf.reduce_sum(tf.losses.mse(real_samples[:, num_categorical_features:],
                                                                   reconstructed_samples[:, total_categories:])),
                       false_fn=lambda: 0.
                       )
    loss += mse_loss
    return loss
