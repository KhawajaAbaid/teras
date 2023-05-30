import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from teras.layers.CTGAN import GeneratorResidualBlock, DiscriminatorBlock
from teras.layers.activations import GumbelSoftmax
from teras.losses.CTGAN import generator_loss, discriminator_loss, generator_dummy_loss
from teras.preprocessing.CTGAN import DataTransformer, DataSampler
from typing import List, Union, Tuple
from functools import partial


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class Generator(keras.Model):
    """
    Generator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        generator_dim: A list or tuple of integers. For each value, a Residual block
            of that dimensionality is added to the generator.
            Defaults to [256, 256].
        data_dim: Dimensionality of the transformed dataset.
        features_meta_data: A dictionary of features meta data.
            Obtained from data_transformer.features_meta_data
    """
    def __init__(self,
                 generator_dim: LIST_OR_TUPLE = [256, 256],
                 data_dim: int = None,
                 features_meta_data: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        assert isinstance(generator_dim, list) or isinstance(generator_dim, tuple),\
            ("generator_dim must be a list or tuple of integers which determines the number of Residual blocks "
            "and the dimensionality of the hidden layer in those blocks.")
        self.generator_dim = generator_dim
        self.data_dim = data_dim
        self.features_meta_data = features_meta_data
        self.generator = models.Sequential()
        for dim in generator_dim:
            self.generator.add(GeneratorResidualBlock(dim))
        self.gumbel_softmax = GumbelSoftmax()
        dense_out = layers.Dense(self.data_dim)
        self.generator.add(dense_out)

    def call(self, inputs):
        # inputs have the shape |z| + |cond|
        # while the outputs will have the shape of equal to (batch_size, transformed_data_dims)
        outputs = []
        interim_outputs = self.generator(inputs)
        continuous_features_relative_indices = self.features_meta_data["continuous"]["relative_indices_all"]

        features_relative_indices_all = self.features_meta_data["relative_indices_all"]
        num_valid_clusters_all = self.features_meta_data["continuous"]["num_valid_clusters_all"]
        cont_i = 0
        cat_i = 0
        num_categories_all = self.features_meta_data["categorical"]["num_categories_all"]
        for i, index in enumerate(features_relative_indices_all):
            # the first k = num_continuous_features are continuous in the data
            if i < len(continuous_features_relative_indices):
                # each continuous features has been transformed into num_valid_clusters + 1 features
                # where the first feature is alpha while the following features are beta components
                alphas = tf.nn.tanh(interim_outputs[:, index])
                alphas = tf.expand_dims(alphas, 1)
                outputs.append(alphas)
                betas = self.gumbel_softmax(interim_outputs[:, index + 1: index + 1 + num_valid_clusters_all[cont_i]])
                outputs.append(betas)
                cont_i += 1
            # elif index in categorical_features_relative_indices:
            else:
                # each categorical feature has been converted into a one hot vector
                # of size num_categories
                ds = self.gumbel_softmax(interim_outputs[:, index: index + num_categories_all[cat_i]])
                outputs.append(ds)
                cat_i += 1
        outputs = tf.concat(outputs, axis=1)
        return outputs


class Discriminator(keras.Model):
    """
    Discriminator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        discriminator_dim: A list or tuple of integers. For each value,
            a Discriminator block of that dimensionality is added to the discriminator.
            Defaults to [256, 256]
        packing_degree: The number of samples concatenated or "packed" together.
            Defaults to 8.
        leaky_relu_alpha: alpha value to use for leaky relu activation
            Defaults to 0.2.
        dropout_rate: Dropout rate to use in the dropout layer
    """
    def __init__(self,
                 discriminator_dim: LIST_OR_TUPLE = [256, 256],
                 packing_degree: int = 8,
                 leaky_relu_alpha: float = 0.2,
                 dropout_rate: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)
        assert isinstance(discriminator_dim, list) or isinstance(discriminator_dim, tuple),\
            ("discriminator_dim must be a list or tuple of integers which determines the number of Discriminator blocks "
            "and the dimensionality of the hidden layer in those blocks.")
        self.discriminator_dim = discriminator_dim
        self.packing_degree = packing_degree
        self.discriminator = keras.models.Sequential()
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate
        for dim in self.discriminator_dim:
            self.discriminator.add(DiscriminatorBlock(dim,
                                                      leaky_relu_alpha=self.leaky_relu_alpha,
                                                      dropout_rate=self.dropout_rate))
        self.discriminator.add(layers.Dense(1))

    @tf.function
    def gradient_penalty(self,
                         real_samples,
                         generated_samples,
                         lambda_=10):
        """
        Calculates the gradient penalty as proposed
        in the paper "Improved Training of Wasserstein GANs"

        Reference(s):
            https://arxiv.org/abs/1704.00028

        Args:
            real_samples:
            generated_samples:
            lambda_:
        """
        batch_size = tf.shape(real_samples)[0]
        dim = tf.shape(real_samples)[1]

        alpha = tf.random.uniform(shape=(batch_size // self.packing_degree, 1, 1))
        alpha = tf.reshape(tf.tile(alpha, [1, self.packing_degree, dim]),
                           (-1, dim))
        interpolated_samples = (alpha * real_samples) + ((1 - alpha) * generated_samples)
        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            y_interpolated = self(interpolated_samples)
        gradients = tape.gradient(y_interpolated, interpolated_samples)
        gradients = tf.reshape(gradients, shape=(-1, self.packing_degree * dim))

        # Calculating gradient penalty
        gradients_norm = tf.norm(gradients)
        gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0)) * lambda_
        return gradient_penalty

    def call(self, inputs):
        inputs_dim = tf.shape(inputs)[1]
        reshaped_inputs = tf.reshape(inputs, shape=(-1, self.packing_degree * inputs_dim))
        return self.discriminator(reshaped_inputs)


class CTGAN(keras.Model):
    """
    This CTGAN implementation provides you with two ways of using CTGAN:
        1. The usual way:
                Specify the parameter values except for
                `generator` and `discriminator`,
                The framework will take care of instantiating
                and building the generator and discriminator models.
        2. The new more flexible way:
                For a finer control, over the building of either generator and discriminator,
                you can import the generator or discriminator model from this `ctgan` module
                or use a complete custom one, build it, and pass it as parameter,
                the framework will ignore the other `discriminator_*` and `generator_*` parameters
                and won't instantiate or build the models.
        NOTE:
            Please note that, you can pass a custom `generator` (or `discriminator`) model
            and specify the `discriminator_*` (or `generator_*`) params, so the framework instantiates
            and builds the  `discriminator` (or `generator`) model the usual way.

    Args:
        latent_dim: Latent dimensionality. Defaults to 128.
        packing_degree: Packing degree - taken from the PacGAN paper.
            The number of samples concatenated or "packed" together.
            It must be a factor of batch_size.
            Defaults to 8.
        use_log_frequency: Whether to calculate probability of values
            by taking log of their frequency in the feature.
            Defaults to True.
        num_discriminator_steps: Number of discriminator steps per training step.
        data_transformer: An instance of DataTransformer class.
        data_sampler: An instance of DataSampler class
        generator: A custom generator model
        discriminator: A custom discriminator model
        generator_dim: A list or tuple of integers. For each value, a Residual block
            of that dimensionality is added to the generator.
            Defaults to [256, 256].
        generator_lr: Learning rate for Generator. Defaults to 1e-3.
        discriminator_dim: A list or tuple of integers. For each value,
            a Discriminator block of that dimensionality is added to the discriminator.
            Defaults to [256, 256]
        discriminator_lr: Learning rate for discriminator.
        gradient_penalty_lambda: Controls the magnitude of the gradient penalty.
            Proposed in the paper: Improved Training of Wasserstein GANs.
            Defaults to 10.
        discriminator_leaky_relu_alpha: Alpha value for leaky relu activation in Discriminator.
        discriminator_dropout_rate: Dropout rate for dropout layer in Discriminator.
    """
    def __init__(self,
                 latent_dim=128,
                 packing_degree=8,
                 log_frequency=True,
                 num_discriminator_steps: int = 1,
                 data_transformer: DataTransformer = None,
                 data_sampler: DataSampler = None,
                 generator: keras.Model = None,
                 discriminator: keras.Model = None,
                 generator_dim: LIST_OR_TUPLE = [256, 256],
                 generator_lr: float = 1e-3,
                 discriminator_dim: LIST_OR_TUPLE = [256, 256],
                 discriminator_lr: float = 1e-3,
                 gradient_penalty_lambda=10,
                 discriminator_leaky_relu_alpha: float = 0.2,
                 discriminator_dropout_rate: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.packing_degree = packing_degree
        self.log_frequency = log_frequency
        self.num_discriminator_steps = num_discriminator_steps
        self.data_sampler = data_sampler
        self.data_transformer = data_transformer
        # If user specifies a custom generator, we won't instantiate or build a generator.
        # All the generator_* params will be ignored.
        self.generator = generator
        if self.generator is None:
            self.generator_dim = generator_dim
            self.generator_lr = generator_lr

        # If user specifies a custom discriminator, we won't instantiate or build a discriminator.
        # All the discriminator_* params will be ignored.
        self.discriminator = discriminator
        if discriminator is None:
            self.discriminator_dim = discriminator_dim
            self.gradient_penalty_lambda = gradient_penalty_lambda
            self.discriminator_leaky_relu_alpha = discriminator_leaky_relu_alpha
            self.discriminator_dropout_rate = discriminator_dropout_rate
            self.discriminator_lr = discriminator_lr
            # instantiate discriminator
            self.discriminator = Discriminator(discriminator_dim=self.discriminator_dim,
                                               leaky_relu_alpha=self.discriminator_leaky_relu_alpha,
                                               dropout_rate=self.discriminator_dropout_rate)
            self.discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=self.generator_lr,
                                                                       beta_1=0.5, beta_2=0.9),
                                       loss=discriminator_loss)
        self.features_meta_data = self.data_transformer.features_meta_data
        self.data_dim = self.features_meta_data["total_transformed_features"]
        # self.built_generator = False
        self.generator = Generator(generator_dim=self.generator_dim,
                                   data_dim=self.data_dim,
                                   features_meta_data=self.features_meta_data)
        self.generator.compile(optimizer=keras.optimizers.Adam(learning_rate=self.generator_lr,
                                                               beta_1=0.5, beta_2=0.9),
                               loss=generator_dummy_loss)
    def call(self, inputs, cond_vector=None):
        generated_samples = self.generator(inputs)
        generated_samples = tf.concat([generated_samples, cond_vector], axis=1)
        y_generated = self.discriminator(generated_samples, training=False)
        return generated_samples, y_generated

    def train_step(self, data):
        real_samples, shuffled_idx, random_features_indices, random_values_indices = data
        self.batch_size = tf.shape(real_samples)[0]

        for _ in range(self.num_discriminator_steps):
            z = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            cond_vector, mask = self.data_sampler.sample_cond_vector_for_training(self.batch_size,
                                                                                  random_features_indices=random_features_indices,
                                                                                  random_values_indices=random_values_indices)
            input_gen = tf.concat([z, cond_vector], axis=1)
            generated_samples = self.generator(input_gen, training=False)
            cond_vector_2 = tf.gather(cond_vector, indices=tf.cast(shuffled_idx, tf.int32))
            generated_samples_cat = tf.concat([generated_samples, cond_vector], axis=1)
            real_samples_cat = tf.concat([real_samples, cond_vector_2], axis=1)

            with tf.GradientTape(persistent=True) as tape:
                y_generated = self.discriminator(generated_samples_cat)
                y_real = self.discriminator(real_samples_cat)
                grad_pen = self.discriminator.gradient_penalty(real_samples_cat, generated_samples_cat)
                loss_disc = self.discriminator.compiled_loss(y_real, y_generated)
            gradients_pen = tape.gradient(grad_pen, self.discriminator.trainable_weights)
            gradients_loss = tape.gradient(loss_disc, self.discriminator.trainable_weights)
            self.discriminator.optimizer.apply_gradients(zip(gradients_pen, self.discriminator.trainable_weights))
            self.discriminator.optimizer.apply_gradients(zip(gradients_loss, self.discriminator.trainable_weights))

        z = tf.random.normal(shape=[self.batch_size, self.latent_dim])
        cond_vector, mask = self.data_sampler.sample_cond_vector_for_training(self.batch_size,
                                                                              random_features_indices=random_features_indices,
                                                                              random_values_indices=random_values_indices)
        # Practically speaking, we don't really need the partial function,
        # but it makes things look more neat
        # generator_partial_loss_fn = partial(generator_loss, mask=mask, features_meta_data=self.features_meta_data)
        input_gen = tf.concat([z, cond_vector], axis=1)
        with tf.GradientTape() as tape:
            tape.watch(cond_vector)
            tape.watch(mask)
            generated_samples, y_generated = self(input_gen, cond_vector=cond_vector)
            loss_gen = generator_loss(generated_samples, y_generated,
                                      cond_vector=cond_vector, mask=mask,
                                      features_meta_data=self.features_meta_data)
            dummy_targets = tf.zeros(shape=(self.batch_size,))
            loss_gen_dummy = self.generator.compiled_loss(dummy_targets, loss_gen)

        gradients = tape.gradient(loss_gen_dummy, self.generator.trainable_weights)
        self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))
        results = {m.name: m.result() for m in self.metrics}
        generator_results = {'generator_'+m.name: m.result() for m in self.generator.metrics}
        results.update(generator_results)
        discriminator_results = {'discriminator_'+m.name: m.result() for m in self.discriminator.metrics}
        results.update(discriminator_results)
        return results

    def generate_new_data(self, num_samples, reverse_transform=True):
        """
        Args:
            num_samples: Number of new samples to generate
            reverse_transform: Whether to reverse transform the generated data to the original data format.
                Defaults to True. If False, the raw generated data will be returned.
        """
        num_steps = num_samples // self.batch_size
        num_steps += 1 if num_samples % self.batch_size != 0 else 0
        generated_samples = []
        for _ in range(num_steps):
            z = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            cond_vector = self.data_sampler.sample_cond_vector_for_generation(self.batch_size)
            input_gen = tf.concat([z, cond_vector], axis=1)
            generated_samples.append(self.generator(input_gen, training=False))
        generated_samples = tf.concat(generated_samples, axis=0)
        generated_samples = generated_samples[:num_samples]

        if reverse_transform:
            generated_samples = self.data_transformer.reverse_transform(x_generated=generated_samples)

        return generated_samples
