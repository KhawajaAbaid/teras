from teras._src import backend
from teras._src.api_export import teras_export


@teras_export("teras.utils.create_gain_dataset")
def create_gain_dataset(x, seed: int = 1337):
    """
    Creates a torch dataloader compatible with the `GAIN` architecture.
    The resultant dataset produces a tuple consisting of a batch of data for
    `generator` and another batch of data for the `discriminator`.

    Args:
        x: Dataset to use for training.
        seed: int, seed to use in shuffling. Defaults to 1337.
    """
    return backend.utils.create_gain_dataset(x, seed)

