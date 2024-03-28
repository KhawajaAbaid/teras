import keras
from keras import ops
from abc import abstractmethod

import numpy as np
from keras.backend import backend
from teras.backend.generic_utils import dataset_type
import time


_BACKEND = backend()


class JAXGAN:
    """
    A base class for building specialized GAN-based architectures in Teras,
    for JAX-backend.

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

        self._built = False
        self._trained = False

    @abstractmethod
    def build(self, input_shape):
        raise NotImplementedError

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(
                    name="generator_optimizer"),
                discriminator_optimizer=keras.optimizers.Adam(
                    name="discriminator_optimizer")
                ):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

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
    def train_step(self, generator_state, discriminator_state, data):
        raise NotImplementedError

    def fit(self, x, epochs: int = 1, verbose: bool = True):
        """
        Minimalist fit method analogous to Keras's `Model.fit` method.

        Args:
            x: Training dataset.
            epochs: Number of epochs to train for.
            verbose: Whether to print outputs.

        Returns:
            A tuple containing the trained `generator_state` and
            `discriminator_state`.
        """
        total_batches = 0
        if dataset_type(x) != "not_supported":
            raise ValueError(
                "Unsupported type for `x`. "
                "It should be tensorflow dataset, pytorch dataloader."
                f"But received {type(x)}")

        # Create initial state
        generator_state = (
            self.generator.trainable_variables,
            self.generator.non_trainable_variables,
            self.generator_optimizer.variables
        )

        discriminator_state = (
            self.discriminator.trainable_variables,
            self.discriminator.non_trainable_variables,
            self.discriminator_optimizer.variables
        )

        for epoch in range(epochs):
            epoch_start_time = time.time()
            for batch_num, batch in enumerate(x):
                if not self._built:
                    if isinstance(batch, tuple):
                        self.build(ops.shape(batch[0]))
                    else:
                        self.build(ops.shape(batch))
                    self._built = True
                batch = np.asarray(batch)
                logs, generator_state, discriminator_state = self.train_step(
                    generator_state,
                    discriminator_state,
                    batch)
                if verbose:
                    epoch_str = f"Epoch {epoch + 1}/{epochs}"
                elapsed_time_str = f"Elapsed {time.time() - epoch_start_time:.2f}s"
                if total_batches > 0 and epoch > 0:
                    batch_str = f"Batch {batch_num + 1}/{total_batches}"
                else:
                    batch_str = f"Batch {batch_num + 1}/?"
                total_batches += 1
                logs_str = (f"generator_loss: "
                            f"{logs['generator_loss']:.4f}   "
                            f"discriminator_loss: "
                            f"{logs['discriminator_loss']::.4f}   "
                            )

                print(
                    f"\r{epoch_str:<15} {elapsed_time_str:<15}"
                    f" {batch_str:<15}"
                    f" {logs_str}",
                    end="")
            print()

            self._trained = True

        return generator_state, discriminator_state
