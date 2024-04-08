import keras
from keras import ops
import numpy as np
from teras._src.api_export import teras_export


@teras_export("teras.losses.saint_constrastive_loss")
def saint_constrastive_loss(
        real,
        augmented,
        temperature: float = 0.7,
        lambda_: float = 0.5):
    """
    Info-NCE inspired contrastive loss for the pretraining objective in
    the SAINT architecture proposed in the paper,
    "SAINT: Improved Neural Networks for Tabular Data".

    Args:
        real: Encodings of the real samples
        augmented: Encodings of the augmented samples
        temperature: float, Temperature value is used in scaling the
            logits. Defaults to 0.7
        lambda_: float, determines how the losses are combined.
            Defaults to 0.5
    """
    batch_size = ops.shape(real)[0]
    labels = ops.arange(batch_size)
    logits_ab = ops.matmul(real, ops.transpose(augmented)) / temperature
    logits_ba = ops.matmul(augmented, ops.transpose(real)) / temperature
    loss_a = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        y_true=labels,
        y_pred=logits_ab)
    loss_b = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        y_true=labels,
        y_pred=logits_ba)
    loss = lambda_ * (loss_a + loss_b) / 2
    return loss


@teras_export("teras.losses.saint_denoising_loss")
def saint_denoising_loss(real,
                         reconstructed,
                         cardinalities: list):
    """
    Since we apply categorical and numerical embedding layers
    separately and then combine them into a new features matrix
    this effectively makes the first k features in the outputs
    categorical (since categorical embeddings are applied first)
    and all other features numerical.
    Here, k = num_categorical_features

    Args:
        real: Samples drawn from the original dataset.
        reconstructed: Samples reconstructed by the reconstruction head.
        cardinalities: list, a list cardinalities of all the features
            in the dataset in the same order as the features' occurrence.
            For numerical features, use the value `0` as indicator at
            the corresponding index.
            You can use the `compute_cardinalities` function from
            `teras.utils` package for this purpose.
    """
    if len(cardinalities) != ops.shape(real)[1]:
        raise AssertionError(
            "`cardinalities` must have the length equal to the number of "
            "features in the dataset. "
            f"Received, len(cardinalities)={len(cardinalities)} and `real`"
            f" with dimensions f{ops.shape(real)[1]}"
        )
    num_features = len(cardinalities)
    categorical_features_idx = np.flatnonzero(cardinalities)
    num_categorical_features = len(categorical_features_idx)
    categorical_cardinalities = np.asarray(cardinalities)[
        categorical_features_idx]
    total_categories = sum(cardinalities)
    loss = 0.
    # The reconstructed samples have dimensions equal to number of
    # categorical features + number of categories in all the categorical
    # features combined.
    # In other words, each categorical feature gets expanded into
    # `number of categories` features due to softmax layer during the
    # reconstruction phase.
    # Since the order they occur in is still the same
    # except for that the first `total_categories` number of features are
    # categorical and following are continuous
    for current_idx, feature_idx in enumerate(categorical_features_idx):
        num_categories = categorical_cardinalities[current_idx]
        loss += keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)(
            # real_samples have the original feature order
            y_true=real[:, feature_idx],
            y_pred=reconstructed[:, current_idx: current_idx + num_categories]
        )
        current_idx += 1
    # Check if there are numerical features -- if so, compute the mse loss
    mse_loss = ops.cond(pred=num_categorical_features < num_features,
                        true_fn=lambda: ops.sum(
                            keras.losses.MeanSquaredError()(
                                real[:, num_categorical_features:],
                                reconstructed[:, total_categories:])),
                        false_fn=lambda: 0.
                        )
    loss += mse_loss
    return loss
