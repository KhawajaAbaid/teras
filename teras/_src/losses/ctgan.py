import keras
from keras import ops
from teras._src.api_export import teras_export


@teras_export("teras.losses.ctgan_generator_loss")
def ctgan_generator_loss(x_generated,
                         y_pred_generated,
                         cond_vectors,
                         mask,
                         metadata):
    """
    Loss for the Generator model in the CTGAN architecture.

    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        x_generated: Samples drawn from the input dataset
        y_pred_generated: Discriminator's output for the generated samples
        cond_vectors: Conditional vectors that are used for and with
            generated samples
        mask: Mask created during the conditional vectors generation step
        metadata: dict, metadata computed during the data transformation step.

    Returns:
        Generator's loss.
    """
    loss = []
    cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=None)
    numerical_features_relative_indices = metadata["numerical"]["relative_indices_all"]
    features_relative_indices_all = metadata["relative_indices_all"]
    num_categories_all = metadata["categorical"]["num_categories_all"]
    # the first k features in the data are numerical which we'll ignore as
    # we're only concerned with the categorical features here
    offset = len(numerical_features_relative_indices)
    for i, index in enumerate(features_relative_indices_all[offset:]):
        logits = x_generated[:, index: index + num_categories_all[i]]
        temp_cond_vector = cond_vectors[:, i: i + num_categories_all[i]]
        labels = ops.argmax(temp_cond_vector, axis=1)
        ce_loss = cross_entropy_loss(y_pred=logits,
                                     y_true=labels
                                     )
        loss.append(ce_loss)
    loss = ops.stack(loss, axis=1)
    loss = ops.sum(loss * ops.cast(mask, dtype="float32")
                   ) / ops.cast(ops.shape(y_pred_generated)[0], dtype="float32")
    loss = -ops.mean(y_pred_generated) * loss
    return loss


@teras_export("teras.losses.ctgan_discriminator_loss")
def ctgan_discriminator_loss(y_pred_real, y_pred_generated):
    """
    Loss for the Discriminator model in the CTGAN architecture.

    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        y_pred_real: Discriminator's output for real samples
        y_pred_generated: Discriminator's output for generated samples

    Returns:
        Discriminator's loss.
    """
    return -(ops.mean(y_pred_real) - ops.mean(y_pred_generated))
