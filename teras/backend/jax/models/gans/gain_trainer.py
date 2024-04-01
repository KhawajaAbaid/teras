import time
import jax
import numpy as np
import keras
from keras import random, ops
from keras.backend import floatx
from teras.backend.generic_utils import dataset_type

# If it works, it works.

_generator: keras.Model = None
_discriminator: keras.Model = None
_generator_optimizer = keras.optimizers.Adam(name="generator_optimizer")
_discriminator_optimizer = keras.optimizers.Adam(name="discriminator_optimizer")
_models_built = False
_optimziers_built = False
_generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
_discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")


def metrics():
    return [_generator_loss_tracker,
            _discriminator_loss_tracker]


def setup(generator,
          discriminator,
          generator_optimizer,
          discriminator_optimizer):
    global _generator
    global _discriminator
    global _generator_optimizer
    global _discriminator_optimizer
    _generator = generator
    _discriminator = discriminator
    _generator_optimizer = generator_optimizer
    _discriminator_optimizer = discriminator_optimizer


def gain_trainer(x, hint_rate: float = 0.9, alpha: float = 100.,
                 seed: int = 1337, epochs: int = 1, verbose: bool = True):
    global _models_built
    global _optimziers_built
    if _generator is None or _discriminator is None:
        raise ValueError("Please call `setup` before using `gain_trainer`")
    total_batches = 0
    if dataset_type(x) == "not_supported":
        raise ValueError(
            "Unsupported type for `x`. "
            "It should be tensorflow dataset, pytorch dataloader."
            f"But received {type(x)}")
    aux_args = (hint_rate, alpha)
    key = jax.random.PRNGKey(seed)
    is_first_iteration = True
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for batch_num, batch in enumerate(x):
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
                    _built = True

                if not _optimziers_built:
                    _generator_optimizer.build(_generator.trainable_variables)
                    _discriminator_optimizer.build(
                        _discriminator.trainable_variables)

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


def compute_discriminator_loss(mask, mask_pred):
    return keras.losses.BinaryCrossentropy()(mask, mask_pred)


def compute_generator_loss(x, x_generated, mask, mask_pred, alpha):
    cross_entropy_loss = keras.losses.CategoricalCrossentropy()(
        mask, mask_pred
    )
    mse_loss = keras.losses.MeanSquaredError()(
        y_true=(mask * x),
        y_pred=(mask * x_generated))
    loss = cross_entropy_loss + alpha * mse_loss
    return loss


def generator_compute_loss_and_updates(
        generator_trainable_vars,
        generator_non_trainable_vars,
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x_gen,
        hint_vectors,
        mask,
        alpha,
        training=False
):
    x_generated, generator_non_trainable_vars = _generator.stateless_call(
        generator_trainable_vars,
        generator_non_trainable_vars,
        ops.concatenate([x_gen, mask], axis=1),
        training=training,
    )
    x_hat = (x_generated * (1 - mask)) + (x_gen * mask)
    mask_pred, _ = (
        _discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            ops.concatenate([x_hat, hint_vectors], axis=1),
            training=training,
        )
    )
    loss = compute_generator_loss(
        x=x_gen,
        x_generated=x_generated,
        mask=mask,
        mask_pred=mask_pred,
        alpha=alpha
    )
    return loss, (generator_non_trainable_vars,)


def discriminator_compute_loss_and_updates(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        generator_trainable_vars,
        generator_non_trainable_vars,
        x_disc,
        hint_vectors,
        mask,
        training=False
):
    x_generated, _ = _generator.stateless_call(
        generator_trainable_vars,
        generator_non_trainable_vars,
        ops.concatenate([x_disc, mask], axis=1),
        training=training,
    )
    x_hat_disc = (x_generated * (1 - mask)) + (x_disc * mask)
    mask_pred, discriminator_non_trainable_vars = (
        _discriminator.stateless_call(
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            ops.concatenate([x_hat_disc, hint_vectors], axis=1),
            training=training,
        )
    )
    loss = compute_discriminator_loss(mask, mask_pred)
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
        hint_rate,
        alpha,
    ) = aux_args

    # data is a tuple of x_generator and x_discriminator batches
    # drawn from the dataset. The reason behind generating two separate
    # batches of data at each step is that it's how GAIN's algorithm works
    x_gen, x_disc = data

    # =========================
    # Train the discriminator
    # =========================
    # Create mask
    mask = 1. - ops.cast(ops.isnan(x_disc), dtype=floatx())
    # replace nans with 0.
    x_disc = ops.where(ops.isnan(x_disc), x1=0., x2=x_disc)
    # Sample noise
    z = random.uniform(shape=ops.shape(x_disc), minval=0., maxval=0.01,
                       seed=key)
    # Sample hint vectors
    hint_vectors = random.binomial(shape=ops.shape(x_disc),
                                   counts=1,
                                   probabilities=hint_rate,
                                   seed=key)
    hint_vectors = hint_vectors * mask
    # Combine random vectors with original data
    x_disc = x_disc * mask + (1 - mask) * z

    disc_grad_fn = jax.value_and_grad(
        discriminator_compute_loss_and_updates,
        has_aux=True,
    )
    (d_loss, (discriminator_non_trainable_vars,)), grads = disc_grad_fn(
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        generator_trainable_vars,
        generator_non_trainable_vars,
        x_disc,
        hint_vectors,
        mask,
        training=True,
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
    # Train the generator
    # =====================
    mask = 1. - ops.cast(ops.isnan(x_gen), dtype=floatx())
    x_gen = ops.where(ops.isnan(x_gen), x1=0., x2=x_gen)
    z = random.uniform(shape=ops.shape(x_gen), minval=0., maxval=0.01,
                       seed=key)
    hint_vectors = random.binomial(shape=ops.shape(x_gen),
                                   counts=1,
                                   probabilities=hint_rate,
                                   seed=key)
    hint_vectors = hint_vectors * mask
    x_gen = x_gen * mask + (1 - mask) * z

    gen_grad_fn = jax.value_and_grad(
        generator_compute_loss_and_updates,
        has_aux=True,
    )
    (g_loss, (generator_non_trainable_vars,)), grads = gen_grad_fn(
        generator_trainable_vars,
        generator_non_trainable_vars,
        discriminator_trainable_vars,
        discriminator_non_trainable_vars,
        x_gen,
        hint_vectors,
        mask,
        alpha,
        training=True,
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
