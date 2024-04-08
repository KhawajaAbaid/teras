import time
import jax
import numpy as np
import keras
from keras import random, ops
from teras._src.backend.generic_utils import dataset_type
from teras._src.losses.ctgan import ctgan_generator_loss, ctgan_discriminator_loss

# If it works, it works.

# Define variables for CTGAN
_generator: keras.Model = None
_discriminator: keras.Model = None
_metadata = None
_latent_dim = None
_seed = None
_generator_optimizer = None
_discriminator_optimizer = None
_models_built = False
_generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
_discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

_initialized: bool = False
_compiled: bool = False


def init(generator: keras.Model, discriminator: keras.Model,
         metadata: dict, latent_dim: int = 128, seed: int = 1337):
    """
    `init` method analogous to `keras.Model.__init__()`.


    Args:
        generator: keras.Model, instance of `CTGANGenerator`
        discriminator: keras.Model, instance of `CTGANDiscrimniator`.
        metadata: dict, A dictionary containing features metadata computed
            during the data transformation step.
            It can be accessed through the `.metadata` property attribute of
            the `CTGANDataTransformer` instance which was used to transform
            the raw input data.
            Note that, this is NOT the same metadata as `features_metadata`,
            which is computed using the `get_metadata_for_embedding` utility
            function from `teras.utils`.
        latent_dim: int, Dimensionality of noise or `z` that serves as
            input to `CTGANGenerator` to generate samples. Defaults to 128.
        seed: int, seed for random ops.
    """
    global _generator
    global _discriminator
    global _metadata
    global _latent_dim
    global _seed
    global _initialized
    _generator = generator
    _discriminator = discriminator
    _metadata = metadata
    _latent_dim = latent_dim
    _seed = seed

    _initialized = True


def compile(generator_optimizer=keras.optimizers.Adam(
    name="generator_optimizer"),
        discriminator_optimizer=keras.optimizers.Adam(
            name="discriminator_optimizer")):
    """
    Compile method analogous to `keras.Model.compile()`.

    Args:
        generator_optimizer: Optimizer for generator.
        discriminator_optimizer: Optimizer for discriminator.
    """
    global _generator_optimizer
    global _discriminator_optimizer
    global _compiled
    if not _initialized:
        raise RuntimeError(
            "Trainer not initialized. Please call `Trainer.init` method "
            "before calling `compile`. "
        )

    _generator_optimizer = generator_optimizer
    _discriminator_optimizer = discriminator_optimizer

    _compiled = True


def build(input_shape):
    """
    `build` method analogous to `keras.Model.build()`.

    Args:
        input_shape: shape of the input dataset
    """
    global _models_built
    global _generator
    global _discriminator
    if not _initialized:
        raise RuntimeError(
            "Trainer not initialized. Please call `Trainer.init` method "
            "before calling `build`. "
        )

    batch_size, input_dim = input_shape
    total_cats = _metadata["categorical"]["total_num_categories"]
    # Generator receives the input of dimensions = data_dim + |cond_vector|
    # where, |cond_vector| = total_num_categories
    discriminator_input_shape = (batch_size, input_dim + total_cats)
    _discriminator.build(discriminator_input_shape)
    # Generator receives the input of dimensions = latent_dim + |cond_vector|
    # where, |cond_vector| = total_num_categories
    generator_input_shape = (batch_size, _latent_dim + total_cats)
    _generator.build(generator_input_shape)

    _models_built = True


def metrics():
    """
    Returns:
        List of metrics.
    """
    return [_generator_loss_tracker,
            _discriminator_loss_tracker]


def fit(x, epochs: int = 1, verbose: bool = True):
    """
    Trainer function for GAIN for JAX backend.

    Args:
        x: Dataset. An instance of `CTGANDataSampler` which can be accessed
            through `teras.preprocessing` package.
        epochs: int, Number of epochs to train.
        verbose: bool, whether to print training logs to console.

    Returns:
        A tuple of trained `generator_state` and `discriminator_state`.
    """
    global _models_built
    global _initialized
    global _compiled
    global _metadata
    global _latent_dim
    global _seed

    if not _initialized:
        raise RuntimeError(
            "Trainer not initialized. Please call `Trainer.init` method "
            "before calling `fit`. "
        )

    if not _compiled:
        raise RuntimeError(
            "Trainer not compiled. Please call `Trainer.compile()` method "
            "before calling `fit`. "
        )

    _optimizers_built = False
    total_batches = 0
    if dataset_type(x) == "not_supported":
        raise ValueError(
            "Unsupported type for `x`. "
            "It should be tensorflow dataset, pytorch dataloader."
            f"But received {type(x)}")
    aux_args = (_metadata, _latent_dim)
    key = jax.random.PRNGKey(_seed)
    is_first_iteration = True
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for batch_num, batch in enumerate(x):
            if isinstance(batch, tuple):
                batch = (np.asarray(elem) for elem in batch)
            else:
                batch = np.asarray(batch)
            if is_first_iteration:
                if not _models_built:
                    if isinstance(batch, tuple):
                        input_shape = ops.shape(batch[0])
                    else:
                        input_shape = ops.shape(batch)
                    input_shape = (input_shape[:-1], input_shape[-1] * 2)
                    _generator.build(input_shape)
                    _discriminator.build(input_shape)
                    _models_built = True

                if not _optimizers_built:
                    _generator_optimizer.build(_generator.trainable_variables)
                    _discriminator_optimizer.build(
                        _discriminator.trainable_variables)
                    _optimizers_built = True

                # Create initial state
                generator_state = (
                    _generator.trainable_variables,
                    _generator.non_trainable_variables,
                    _generator_optimizer.variables
                )

                discriminator_state = (
                    _discriminator.trainable_variables,
                    _discriminator.non_trainable_variables,
                    _discriminator_optimizer.variables
                )

                is_first_iteration = False

            # print("\nSTATES:\n"
            #       f"\tGenerator: {generator_state}\n"
            #       f"\tDiscriminator: {discriminator_state}"
            #       )
            logs, generator_state, discriminator_state = train_step(
                generator_state,
                discriminator_state,
                batch,
                aux_args,
                jax.random.split(key, 1)[0]
            )
            if verbose:
                epoch_str = f"Epoch {epoch + 1}/{epochs}"
                elapsed_time_str = (f"Elapsed "
                                    f"{time.time() - epoch_start_time:.2f}s")
                if total_batches > 0 and epoch > 0:
                    batch_str = f"Batch {batch_num + 1}/{total_batches}"
                else:
                    batch_str = f"Batch {batch_num + 1}/?"
                    total_batches += 1
                logs_str = (f"generator_loss: "
                            f"{logs['generator_loss']:.4f}   "
                            f"discriminator_loss: "
                            f"{logs['discriminator_loss']:.4f}   "
                            )
                print(
                    f"\r{epoch_str:<15} {elapsed_time_str:<15} {batch_str:<15}"
                    f" {logs_str}",
                    end="")
        print()
        _trained = True

    return generator_state, discriminator_state


def generator_compute_loss_and_updates(
        generator_trainable_vars,
        generator_non_trainable_vars,
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        noise,
        cond_vectors,
        mask,
        metadata
):
    x_generated = _generator.stateless_call(
        generator_trainable_vars,
        generator_non_trainable_vars,
        noise)
    x_generated = ops.concatenate(
        [x_generated, cond_vectors], axis=1)
    y_pred_generated = _discriminator.stateless_call(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x_generated)
    loss = ctgan_generator_loss(x_generated, y_pred_generated,
                                cond_vectors=cond_vectors, mask=mask,
                                metadata=metadata)
    return loss, (generator_non_trainable_vars,)


def discriminator_compute_grad_penalty_and_updates(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x,
        x_generated,
):
    y_pred_generated = _discriminator.stateless_call(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x_generated)
    y_pred_real = _discriminator.stateless_call(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x)
    grad_pen = _discriminator.gradient_penalty(x, x_generated)
    return grad_pen, (discriminator_non_trainable_vars,)


def discriminator_compute_loss_and_updates(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x,
        x_generated,
):
    y_pred_generated = _discriminator.stateless_call(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x_generated)
    y_pred_real = _discriminator.stateless_call(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x)
    loss = ctgan_discriminator_loss(y_pred_real, y_pred_generated)
    return loss, (discriminator_non_trainable_vars,)


@jax.jit
def train_step(generator_state, discriminator_state, data, aux_args, key):
    (
        generator_trainable_vars,
        generator_non_trainable_vars,
        generator_optimizer_vars,
    ) = generator_state

    (
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        discriminator_optimizer_vars,
    ) = discriminator_state

    (
        metadata,
        latent_dim,
    ) = aux_args

    x, cond_vectors_real, cond_vectors, mask = data
    batch_size = ops.shape(x)[0]

    # =========================
    # Discriminator train step
    # =========================
    z = random.normal(shape=[batch_size, latent_dim],
                      seed=key)
    input_gen = ops.concatenate([z, cond_vectors], axis=1)
    x_generated, _ = _generator.stateless_call(
        generator_trainable_vars,
        generator_non_trainable_vars,
        input_gen)
    x_generated = ops.concatenate(
        [x_generated, cond_vectors], axis=1)
    x = ops.concatenate([x, cond_vectors_real],
                        axis=1)
    # w.r.t. gradient penalty
    disc_grad_fn_pen = jax.value_and_grad(
        discriminator_compute_grad_penalty_and_updates,
        has_aux=True)
    (grad_pen, (discriminator_non_trainable_vars,)), grads = disc_grad_fn_pen(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x,
        x_generated
    )
    (
        discriminator_trainable_vars,
        discriminator_optimizer_vars
    ) = _discriminator_optimizer.stateless_apply(
        discriminator_optimizer_vars,
        grads,
        discriminator_trainable_vars
    )
    # w.r.t. loss
    disc_grad_fn_loss = jax.value_and_grad(
        discriminator_compute_loss_and_updates,
        has_aux=True)
    (d_loss, (discriminator_non_trainable_vars,)), grads = disc_grad_fn_loss(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x,
        x_generated
    )
    (
        discriminator_trainable_vars,
        discriminator_optimizer_vars
    ) = _discriminator_optimizer.stateless_apply(
        discriminator_optimizer_vars,
        grads,
        discriminator_trainable_vars
    )

    discriminator_state = (
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        discriminator_optimizer_vars
    )

    # =====================
    # Generator train step
    # =====================
    z = random.normal(shape=[batch_size, latent_dim],
                      seed=jax.random.split(key, 1)[0])
    input_gen = ops.concatenate([z, cond_vectors], axis=1)
    gen_grad_fn = jax.value_and_grad(
        generator_compute_loss_and_updates,
        has_aux=True,
    )
    (g_loss, (generator_non_trainable_vars,)), grads = gen_grad_fn(
        generator_trainable_vars,
        generator_non_trainable_vars,
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        input_gen,
        cond_vectors,
        mask,
        metadata
    )

    (
        generator_trainable_vars,
        generator_optimizer_vars
    ) = _generator_optimizer.stateless_apply(
        generator_optimizer_vars,
        grads,
        generator_trainable_vars
    )

    generator_state = (
        generator_trainable_vars,
        generator_non_trainable_vars,
        generator_optimizer_vars
    )

    # Update custom tracking metrics
    _generator_loss_tracker.update_state(g_loss)
    _discriminator_loss_tracker.update_state(d_loss)

    logs = {m.name: m.result() for m in metrics()}
    return logs, generator_state, discriminator_state
