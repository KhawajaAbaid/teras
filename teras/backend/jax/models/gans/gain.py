# ===================
# THIS DOES NOT WORK!
# ===================

import jax
import keras
from keras import random, ops
from keras.backend import floatx

from teras.backend.jax.models.gans.jax_gan import JAXGAN


@jax.tree_util.register_pytree_node_class
class GAIN(JAXGAN):
    raise ImportError(
        "JAX backend doesn't provide a MODEL implementation for `GAIN`. "
        "Rather it offers `GAINTrainer` based on functional approach which "
        "can be accessed through `teras.trainers` package. \n"
        "Alternatively, if you really want to use a model based implementation "
        "for `CTGAN`, you can use the `tensorflow` or `torch` backend."
    )
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 hint_rate: float = 0.9,
                 alpha: float = 100.,
                 seed: int = 1337
                 ):
        super().__init__(generator=generator,
                         discriminator=discriminator)
        self.generator = generator
        self.discriminator = discriminator
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.seed = seed
        self.dtype = floatx()

        # Loss trackers
        self.loss_tracker = keras.metrics.Mean(
            name="loss")
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(
            name="discriminator_loss")

    # def _create_seed_gen(self, seed):
    #     self._seed_gen = seed

    @property
    def metrics(self):
        _metrics = [self.loss_tracker,
                    self.generator_loss_tracker,
                    self.discriminator_loss_tracker,
                    ]
        return _metrics

    def build(self, input_shape):
        # Inputs received by each generator and discriminator have twice the
        # dimensions of original inputs
        input_shape = (input_shape[:-1], input_shape[-1] * 2)
        self.generator.build(input_shape)
        self.discriminator.build(input_shape)

    def compute_discriminator_loss(self, mask, mask_pred):
        return keras.losses.BinaryCrossentropy()(mask, mask_pred)

    def compute_generator_loss(self, x, x_generated, mask, mask_pred, alpha):
        cross_entropy_loss = keras.losses.CategoricalCrossentropy()(
            mask, mask_pred
        )
        mse_loss = keras.losses.MeanSquaredError()(
            y_true=(mask * x),
            y_pred=(mask * x_generated))
        loss = cross_entropy_loss + alpha * mse_loss
        return loss

    def generator_compute_loss_and_updates(
            self,
            generator_trainable_vars,
            generator_non_trainable_vars,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            x_gen,
            hint_vectors,
            mask,
            training=False
    ):
        x_generated, generator_non_trainable_vars = self.generator.stateless_call(
            generator_trainable_vars,
            generator_non_trainable_vars,
            ops.concatenate([x_gen, mask], axis=1),
            training=training,
        )
        x_hat = (x_generated * (1 - mask)) + (x_gen * mask)
        mask_pred, _ = (
            self.discriminator.stateless_call(
                discriminator_trainable_vars,
                discriminator_non_trainable_vars,
                ops.concatenate([x_hat, hint_vectors], axis=1),
                training=training,
            )
        )
        loss = self.compute_generator_loss(
            x=x_gen,
            x_generated=x_generated,
            mask=mask,
            mask_pred=mask_pred,
            alpha=self.alpha
        )
        return loss, (generator_non_trainable_vars,)

    def discriminator_compute_loss_and_updates(
            self,
            discriminator_trainable_vars,
            discriminator_non_trainable_vars,
            generator_trainable_vars,
            generator_non_trainable_vars,
            x_disc,
            hint_vectors,
            mask,
            training=False
    ):
        x_generated, _ = self.generator.stateless_call(
            generator_trainable_vars,
            generator_non_trainable_vars,
            ops.concatenate([x_disc, mask], axis=1),
            training=training,
        )
        x_hat_disc = (x_generated * (1 - mask)) + (x_disc * mask)
        mask_pred, discriminator_non_trainable_vars = (
            self.discriminator.stateless_call(
                discriminator_trainable_vars,
                discriminator_non_trainable_vars,
                ops.concatenate([x_hat_disc, hint_vectors], axis=1),
                training=training,
            )
        )
        loss = self.compute_discriminator_loss(mask, mask_pred)
        return loss, (discriminator_non_trainable_vars,)

    @jax.jit
    def train_step(self, generator_state, discriminator_state, data):
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
                           seed=1337)
        # Sample hint vectors
        hint_vectors = random.binomial(shape=ops.shape(x_disc),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=1337)
        hint_vectors = hint_vectors * mask
        # Combine random vectors with original data
        x_disc = x_disc * mask + (1 - mask) * z

        disc_grad_fn = jax.value_and_grad(
            self.discriminator_compute_loss_and_updates,
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
        ) = self.discriminator_optimizer.stateless_apply(
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
                           seed=1337)
        hint_vectors = random.binomial(shape=ops.shape(x_gen),
                                       counts=1,
                                       probabilities=self.hint_rate,
                                       seed=1337)
        hint_vectors = hint_vectors * mask
        x_gen = x_gen * mask + (1 - mask) * z

        gen_grad_fn = jax.value_and_grad(
            self.generator_compute_loss_and_updates,
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
            training=True,
        )

        (
            generator_trainable_vars,
            generator_optimizer_vars
        ) = self.discriminator_optimizer.stateless_apply(
            generator_optimizer_vars,
            grads,
            generator_optimizer_vars
        )

        generator_state = (
            generator_trainable_vars,
            generator_non_trainable_vars,
            generator_optimizer_vars
        )

        # Update custom tracking metrics
        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)

        logs = {m.name: m.result() for m in self.metrics}
        return logs, generator_state, discriminator_state

    def tree_flatten(self):
        generator_flat, generator_def = jax.tree_util.tree_flatten(
            self.generator)
        discriminator_flat, discriminator_def = jax.tree_util.tree_flatten(
            self.discriminator)

        children = (generator_flat, discriminator_flat,
                    self.hint_rate, self.alpha, self.seed)

        aux_data = (generator_def, discriminator_def)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        generator_def, discriminator_def = aux_data
        generator_flat = children[0]
        generator = jax.tree_util.tree_unflatten(generator_def,
                                                 generator_flat)
        generator = generator
        discriminator_flat = children[1]
        discriminator = jax.tree_util.tree_unflatten(discriminator_def,
                                                     discriminator_flat)
        discriminator = discriminator
        hint_rate = children[2]
        alpha = children[3]
        seed = children[4]
        c = cls(generator=generator, discriminator=discriminator,
                hint_rate=hint_rate, alpha=alpha, seed=seed)
        return c
