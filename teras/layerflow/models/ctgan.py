import tensorflow as tf
from tensorflow import keras
from teras.losses.ctgan import generator_loss, discriminator_loss
from tqdm import tqdm


@keras.saving.register_keras_serializable(package="keras.layerflow.models")
class CTGAN(keras.Model):
    """
    CTGAN model with LayerFlow design.
    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        generator: ``keras.Model``,
            An instance of ``CTGANGenerator`` or any other keras model that can work in
            its place.
            You can import the ``CTGANGenerator`` as follows,
                >>> from teras.models import CTGANGenerator

        discriminator: ``keras.Model``,
            An instance of ``CTGANDiscriminator`` or any other keras model that can work in
            its place.
            You can import the ``CTGANDiscriminator`` as follows,
                >>> from teras.models import CTGANDiscriminator

        num_discriminator_steps: ``int``, default 1,
            Number of discriminator training steps per ``CTGAN`` training step.

        latent_dim: ``int``, default 128,
            Dimensionality of noise or ``z`` that serves as input to ``CTGANGenerator``
            to generate samples.
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 num_discriminator_steps: int = 1,
                 latent_dim: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.num_discriminator_steps = num_discriminator_steps
        self.latent_dim = latent_dim

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def compile(self,
                generator_optimizer=keras.optimizers.Adam(learning_rate=1e-3,
                                                          beta_1=0.5, beta_2=0.9),
                discriminator_optimizer=keras.optimizers.Adam(learning_rate=1e-3,
                                                              beta_1=0.5, beta_2=0.9),
                generator_loss=generator_loss,
                discriminator_loss=discriminator_loss,
                **kwargs
                ):
        super().compile(**kwargs)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def call(self, inputs):
        generated_samples = self.generator(inputs)
        return generated_samples

    def train_step(self, data):
        # real_samples, shuffled_idx, random_features_indices, random_values_indices = data
        real_samples, cond_vectors_real, cond_vectors, mask = data
        self.batch_size = tf.shape(real_samples)[0]

        for _ in range(self.num_discriminator_steps):
            z = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            # cond_vector, mask = self.data_sampler.sample_cond_vector_for_training(self.batch_size,
            #                                                                       random_features_indices=random_features_indices,
            #                                                                       random_values_indices=random_values_indices)
            input_gen = tf.concat([z, cond_vectors], axis=1)
            generated_samples = self.generator(input_gen, training=False)
            # cond_vector_2 = tf.gather(cond_vector, indices=tf.cast(shuffled_idx, tf.int32))
            generated_samples = tf.concat([generated_samples, cond_vectors], axis=1)
            real_samples = tf.concat([real_samples, cond_vectors_real], axis=1)

            with tf.GradientTape(persistent=True) as tape:
                y_generated = self.discriminator(generated_samples)
                y_real = self.discriminator(real_samples)
                grad_pen = self.discriminator.gradient_penalty(real_samples, generated_samples)
                loss_disc = self.discriminator_loss(y_real, y_generated)
            gradients_pen = tape.gradient(grad_pen, self.discriminator.trainable_weights)
            gradients_loss = tape.gradient(loss_disc, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(zip(gradients_pen, self.discriminator.trainable_weights))
            self.discriminator_optimizer.apply_gradients(zip(gradients_loss, self.discriminator.trainable_weights))

        z = tf.random.normal(shape=[self.batch_size, self.latent_dim])
        # cond_vector, mask = self.data_sampler.sample_cond_vector_for_training(self.batch_size,
        #                                                                       random_features_indices=random_features_indices,
        #                                                                       random_values_indices=random_values_indices)
        # Practically speaking, we don't really need the partial function,
        # but it makes things look more neat
        # generator_partial_loss_fn = partial(generator_loss, mask=mask, metadata=self.metadata)
        input_gen = tf.concat([z, cond_vectors], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(cond_vectors)
            tape.watch(mask)
            # generated_samples, y_generated = self(input_gen, cond_vector=cond_vector)
            generated_samples = self(input_gen)
            generated_samples = tf.concat([generated_samples, cond_vectors], axis=1)
            y_generated = self.discriminator(generated_samples, training=False)
            loss_gen = self.generator_loss(generated_samples, y_generated,
                                           cond_vectors=cond_vectors, mask=mask,
                                           metadata=self.metadata)
            # dummy_targets = tf.zeros(shape=(self.batch_size,))
            # loss_gen_dummy = self.generator.compiled_loss(dummy_targets, loss_gen)
        gradients = tape.gradient(loss_gen, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        self.generator_loss_tracker.update_state(loss_gen)
        self.discriminator_loss_tracker.update_state(loss_disc)

        results = {m.name: m.result() for m in self.metrics}
        generator_results = {'generator_' + m.name: m.result() for m in self.generator.metrics}
        results.update(generator_results)
        discriminator_results = {'discriminator_' + m.name: m.result() for m in self.discriminator.metrics}
        results.update(discriminator_results)
        return results

    def generate(self,
                 num_samples: int,
                 data_sampler,
                 data_transformer=None,
                 reverse_transform: bool = False,
                 batch_size: int = None):
        """
        Generates new samples using the trained generator.

        Args:
            num_samples: ``int``,
                Number of new samples to generate

            data_sampler:
                Instance of the ``CTGANDataSampler`` class used in preparing
                the tensorflow dataset for training.

            data_transformer:
                Instance of ``CTGANDataTransformer`` class that was used to preprocess the raw data.
                This is required only if the ``reverse_transform`` is set to True.

            reverse_transform: ``bool``, default False,
                Whether to reverse transform the generated data to the original data format.
                If False, the raw generated data will be returned, which you can then manually
                transform into original data format by utilizing ``CTGANDataTransformer`` instance's
                ``reverse_transform`` method.

            batch_size: ``int``, default None.
                If a value is passed, samples will be generated in batches
                where ``batch_size`` determines the size of each batch.
                If ``None``, all ``num_samples`` will be generated at once.
                Note that, if the number of samples to generate aren't huge
                or you know your hardware can handle to generate all samples at once,
                you can leave the value to None, otherwise it is recommended to specify
                a value for batch size.
        """
        if batch_size is None:
            batch_size = num_samples
        num_steps = num_samples // batch_size
        num_steps += 1 if num_samples % batch_size != 0 else 0
        generated_samples = []
        for _ in tqdm(range(num_steps), desc="Generating Data"):
            z = tf.random.normal(shape=[batch_size, self.latent_dim])
            cond_vector = data_sampler.sample_cond_vectors_for_generation(batch_size)
            input_gen = tf.concat([z, cond_vector], axis=1)
            generated_samples.append(self.generator(input_gen, training=False))
        generated_samples = tf.concat(generated_samples, axis=0)
        generated_samples = generated_samples[:num_samples]

        if reverse_transform:
            if data_transformer is None:
                raise ValueError("""To reverse transform the raw generated data, ``data_transformer`` must not be None.
                         Please pass the instance of ``CTGANDataTransformer`` class that was used to transform the
                         input data. Or alternatively, you can set ``reverse_transform`` to False, and later
                         manually reverse transform the generated raw data to original format by utilizing the 
                         ``CTGANDataTransformer`` instance's ``reverse_transform`` method.""")
            generated_samples = data_transformer.reverse_transform(x_generated=generated_samples)

        return generated_samples

    def get_config(self):
        config = super().get_config()
        config.update({'generator': keras.layers.serialize(self.generator),
                       'discriminator': keras.layers.serialize(self.generator),
                       'num_discriminator_steps': self.num_discriminator_steps,
                       'latent_dim': self.latent_dim,
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        generator = keras.layers.deserialize(config.pop("generator"))
        discriminator = keras.layers.deserialize(config.pop("discriminator"))
        return cls(generator=generator,
                   discriminator=discriminator,
                   **config)
