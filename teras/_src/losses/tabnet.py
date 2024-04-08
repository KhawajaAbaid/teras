from keras import ops
from keras.backend import floatx
from teras._src.api_export import teras_export


@teras_export("teras.losses.tabnet_reconstruction_loss")
def tabnet_reconstruction_loss(real=None,
                               reconstructed=None,
                               mask=None):
    """
    Reconstruction loss for TabNet Pretrainer mode as proposed by
    Sercan et al. in the paper,
    "TabNet: Attentive Interpretable Tabular Learning"

     Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        real: Samples drawn from the input dataset
        reconstructed: Samples reconstructed by the decoder
        mask: Mask that indicates the missing-ness of features in a sample

    Returns:
        Reconstruction loss for TabNet Pretraining.
    """
    nominator_part = (reconstructed - real) * mask
    real_samples_population_std = ops.std(ops.cast(real, dtype=floatx()))
    # divide
    x = nominator_part / real_samples_population_std
    # Calculate L2 norm
    loss = ops.sqrt(ops.sum(ops.square(x)))
    return loss
