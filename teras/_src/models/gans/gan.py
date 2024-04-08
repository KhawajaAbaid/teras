import time
from abc import abstractmethod

import keras
from keras.backend import backend
from teras._src.backend.generic_utils import dataset_type

from teras._src.api_export import teras_export

_BACKEND = backend()


@teras_export("teras.models.GAN")
class GAN:
    """
    A base class for building specialized GAN-based architectures in Teras.

    Args:
        generator: An instance of `keras.Model` that would serve as the
            generator for the given GAN architecture.
        discriminator: An instance of `keras.Model` that would serve as
            the discriminator for the given GAN architecture.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model
                 ):
        self.generator = generator
        self.discriminator = discriminator

        self._trained = False

    @property
    def trained_generator(self):
        if not self._trained:
            raise AssertionError(
                "The generator has not yet been trained. Please train the "
                "GAN network using the `fit` method before accessing the "
                "`trained_generator` attribute."
            )
        return self.generator

    @property
    def trained_discriminator(self):
        if not self._trained:
            raise AssertionError(
                "The discriminator has not yet been trained. Please train the "
                "GAN network using the `fit` method before accessing the "
                "`trained_discriminator` attribute."
            )
        return self.discriminator

    @abstractmethod
    def generator_train_step(self, data):
        raise NotImplementedError

    @abstractmethod
    def discriminator_train_step(self, data):
        raise NotImplementedError

    def fit(self, x, batch_size: int = 128, epochs: int = 1,
            verbose: bool = True, shuffle: bool = True):
        """
        Fit method analogous to Keras's `Model.fit` method

        Args:
            x: Training dataset.
            batch_size: Batch size. Pass `None` to treat the whole dataset
                as one single batch.
            epochs: Number of epochs to train for.
            verbose: Whether to print outputs.
            shuffle: Whether to shuffle data. Only applicable when numpy
                ndarray is passed as the `x` argument.

        Returns:
            A dictionary containing the history of training, analogous
            to the history object returned by Keras's `fit` method.
        """

        # TODO:
        #   1. JAX ecosystem uses tf.data. Convert tf tensors to numpy
        #      arrays before passing to the model when on JAX backend.
        #   2. Create batches of data manually when passed a numpy ndarray
        #   3. Return a history object like Keras's fit method does

        total_batches = 0
        if dataset_type(x) == "not_supported":
            raise ValueError(
                "Unsupported type for `x`. "
                "It should be tensorflow dataset, pytorch dataloader or a "
                f"numpy ndarray. But received {type(x)}")
        elif dataset_type(x) == "numpy_ndarray":
            # TODO
            pass

        for epoch in range(epochs):
            epoch_start_time = time.time()
            for batch_num, batch in enumerate(x):
                generator_loss = self.discriminator_train_step(batch)
                discriminator_loss = self.generator_train_step(batch)

                if verbose:
                    epoch_str = f"Epoch {epoch + 1}/{epochs}"
                    elapsed_time_str = f"Elapsed {time.time() - epoch_start_time:.2f}s"
                    if total_batches > 0 and epoch > 0:
                        batch_str = f"Batch {batch_num + 1}/{total_batches}"
                    else:
                        batch_str = f"Batch {batch_num + 1}/?"
                        total_batches += 1
                    logs_str = f"generator_loss: {generator_loss:.4f}   " \
                               f"discriminator_loss: " \
                               f"{discriminator_loss:.4f}   "

                    print(
                        f"\r{epoch_str:<15} {elapsed_time_str:<15}"
                        f" {batch_str:<15}"
                        f" {logs_str}",
                        end="")
                print()
