import keras
from keras import random, ops
from keras.backend import backend
from teras._src.utils import clean_reloaded_config_data


class BaseCTGAN(keras.Model):
    """
    Base CTGAN model class.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 metadata: dict,
                 latent_dim: int = 128,
                 seed: int = 1337,
                 **kwargs):
        if not backend() == "jax":
            # Don't call super() with JAX backend as `JAXGAN` does that
            # already!
            super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.metadata = metadata
        self.latent_dim = latent_dim
        self.seed = seed
        self._seed_gen = random.SeedGenerator(self.seed)

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(
            name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(
            name="discriminator_loss")

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(
                    learning_rate=1e-3,
                    beta_1=0.5, beta_2=0.9,
                    name="generator_optimizer"
                ),
                discriminator_optimizer=keras.optimizers.Adam(
                    learning_rate=1e-3,
                    beta_1=0.5, beta_2=0.9,
                    name="discriminator_optimizer"
                ),
                **kwargs
                ):
        super().compile(**kwargs)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def build(self, input_shape):
        batch_size, input_dim = input_shape
        total_cats = self.metadata["categorical"]["total_num_categories"]
        # Generator receives the input of dimensons = data_dim + |cond_vector|
        # where, |cond_vector| = total_num_categories
        input_shape = (batch_size, input_dim + total_cats)
        self.discriminator.build(input_shape)
        # Generator receives the input of dimensons = latent_dim + |cond_vector|
        # where, |cond_vector| = total_num_categories
        input_shape = (batch_size, self.latent_dim + total_cats)
        self.generator.build(input_shape)

    @property
    def metrics(self):
        return [self.generator_loss_tracker,
                self.discriminator_loss_tracker]

    @staticmethod
    def compute_generator_loss(x_generated,
                               y_pred_generated,
                               cond_vectors,
                               mask,
                               metadata):
        """
        Loss for the Generator model in the CTGAN architecture.

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
        continuous_features_relative_indices = metadata["continuous"]["relative_indices_all"]
        features_relative_indices_all = metadata["relative_indices_all"]
        num_categories_all = metadata["categorical"]["num_categories_all"]
        # the first k features in the data are continuous which we'll ignore as
        # we're only concerned with the categorical features here
        offset = len(continuous_features_relative_indices)
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

    @staticmethod
    def compute_disciriminator_loss(y_pred_real, y_pred_generated):
        """
        Loss for the Discriminator model in the CTGAN architecture.

        Args:
            y_pred_real: Discriminator's output for real samples
            y_pred_generated: Discriminator's output for generated samples

        Returns:
            Discriminator's loss.
        """
        return -(ops.mean(y_pred_real) - ops.mean(y_pred_generated))

    def test_step(self, data):
        x, cond_vectors_real, cond_vectors, mask = data
        batch_size = ops.shape(x)[0]

        # =========================
        # Discriminator test step
        # =========================
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)
        x_generated = self.generator(input_gen)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        x = ops.concatenate([x, cond_vectors_real],
                            axis=1)

        y_pred_generated = self.discriminator(x_generated)
        y_pred_real = self.discriminator(x)
        loss_disc = self.compute_disciriminator_loss(
            y_pred_real,
            y_pred_generated)

        # =====================
        # Generator test step
        # =====================
        z = random.normal(shape=[batch_size, self.latent_dim],
                          seed=self._seed_gen)
        input_gen = ops.concatenate([z, cond_vectors], axis=1)

        x_generated = self.generator(input_gen)
        x_generated = ops.concatenate(
            [x_generated, cond_vectors], axis=1)
        y_pred_generated = self.discriminator(x_generated)
        loss_gen = self.compute_generator_loss(
            x_generated, y_pred_generated,
            cond_vectors=cond_vectors, mask=mask,
            metadata=self.metadata)

        self.generator_loss_tracker.update_state(loss_gen)
        self.discriminator_loss_tracker.update_state(loss_disc)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def predict_step(self, generator_input):
        generated_samples = self.generator(generator_input)
        return generated_samples

    def get_config(self):
        config = super().get_config()
        config.update({
            'generator': keras.layers.serialize(self.generator),
            'discriminator': keras.layers.serialize(self.generator),
            'latent_dim': self.latent_dim,
            'seed': self.seed,
        }
        )
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.layers.deserialize(config.pop("generator"))
        discriminator = keras.layers.deserialize(config.pop("discriminator"))
        metadata = clean_reloaded_config_data(config.pop("metadata"))
        return cls(generator=generator,
                   discriminator=discriminator,
                   metadata=metadata,
                   **config)
