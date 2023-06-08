import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from teras.losses.gain import generator_loss, discriminator_loss
from typing import List, Tuple, Union


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class Generator(keras.Model):
    """
    Generator model for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        units_hidden: `list` | `tuple`, A list or tuple of units.
            For each element, a new hidden layer will be added,
            with units equal to that element value.
            In official implementation, for each layer,
            `units` is equal to the data dimensions,
            and the number of hidden layers is 2.
            So by default we pass [data_dim, data_dim].
        activation_hidden: default "relu", Activation function to use for hidden layers.
        activation_out: default "sigmoid", Activation function to use for the output layer.
        kernel_initializer: default "glorot_normal", Initializer for weights of layers within generator.

    Example:
        ```python

        # Instantiate Generator with default settings
        generator = Generator()

        # Sample noise to generate samples from
        z = tf.random.normal([512, 18])

        # Generate samples
        generated_samples = generator(z)
        ```
    """
    def __init__(self,
                 units_hidden: LIST_OR_TUPLE = None,
                 activation_hidden="relu",
                 activation_out="sigmoid",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.units_hidden = units_hidden
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        self.kernel_initializer = kernel_initializer

        self.hidden_block = models.Sequential(name="generator_hidden_block")

    def build(self, input_shape):
        # inputs is the concatenation of `mask` and `original data`
        # and `mask` has the same dimensions as `data`, so we halve the
        # `input_shape[1]` here to get the `original data` dimensions
        data_dim = input_shape[1] // 2
        if self.units_hidden is None:
            self.units_hidden = [data_dim] * 2
        for units in self.units_hidden:
            self.hidden_block.add(
                layers.Dense(units,
                             activation=self.activation_hidden,
                             kernel_initializer=self.kernel_initializer)
            )
        self.dense_out = layers.Dense(units=data_dim,
                                      activation=self.activation_out,
                                      kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        # inputs is the concatenation of mask and original data
        outputs = self.hidden_block(inputs)
        outputs = self.dense_out(outputs)
        return outputs

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({'units_hidden': self.hidden_dims,
                       'activation_hidden': self.activation_hidden,
                       'activation_out': self.activation_out,
                       'kernel_initializer': self.kernel_initializer,
                       })
        return config


class Discriminator(keras.Model):
    """
    Discriminator model for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        units_hidden: `list` | `tuple`, A list of units.
            For each element, a new hidden layer will be added,
            with units equal to that element value.
            In official implementation, for each layer,
            `units` is equal to the data dimensions,
            and the number of hidden layers is 2.
            So by default we pass [data_dim, data_dim].
        activation_hidden: default "relu", Activation function
            to use for hidden layers.
        activation_out: default "sigmoid", Activation function
            to use for the output layer.
        kernel_initializer: default "glorot_normal",
            Initializer for weights of layers within discriminator.

    Example:
        ```python

        # Instantiate Generator with default settings
        generator = Generator()

        # Instantiate Discriminator with default settings
        discriminator = Discriminator()

        # Sample noise to generate samples from
        z = tf.random.normal([512, 18])

        # Generate samples
        generated_samples = generator(z)

        # Predict using discriminator
        y_pred = discriminator(generated_samples)
        ```
    """
    def __init__(self,
                 units_hidden: LIST_OR_TUPLE = None,
                 activation_hidden="relu",
                 activation_out="sigmoid",
                 kernel_initializer="glorot_normal",
                 **kwargs):
        super().__init__(**kwargs)
        self.units_hidden = units_hidden
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out
        self.kernel_initializer = kernel_initializer

        self.hidden_block = models.Sequential(name="discriminator_hidden_block")

    def build(self, input_shape):
        # inputs is the concatenation of `hint` and original data
        # and `hint` has the same dimensions as data, so we halve the
        # input_shape[1] here to get the original data dimensions
        data_dim = input_shape[1] // 2
        if self.units_hidden is None:
            self.units_hidden = [data_dim] * 2
        for units in self.units_hidden:
            self.hidden_block.add(layers.Dense(units,
                                               activation=self.activation_hidden,
                                               kernel_initializer=self.kernel_initializer))
        self.dense_out = layers.Dense(units=data_dim,
                                      activation=self.activation_out,
                                      kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        # inputs is the concatenation of `hint` and `original data`
        outputs = self.hidden_block(inputs)
        outputs = self.dense_out(outputs)
        return outputs

    def get_config(self):
        config = super(Discriminator, self).get_config()
        config.update({'units_hidden': self.units_hidden,
                       'activation_hidden': self.activation_hidden,
                       'activation_out': self.activation_out,
                       'kernel_initializer': self.kernel_initializer})
        return config


class GAIN(keras.Model):
    """
    GAIN is a missing data imputation model based on GANs.
    This is an implementation of the GAIN architecture
    proposed by Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    In GAIN, the generator observes some components of
    a real data vector, imputes the missing components
    conditioned on what is actually observed, and
    outputs a completed vector.
    The discriminator then takes a completed vector
    and attempts to determine which components
    were actually observed and which were imputed.
    It also utilizes a novel hint mechanism, which
    ensures that generator does in fact learn to generate
    samples according to the true data distribution.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        hint_rate: Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Must be between 0. and 1.
            Hint vectors ensure that generated samples follow the
            underlying data distribution.
        alpha: Hyper parameter for the generator loss computation that
            controls how much weight should be given to the MSE loss.
            Precisely, `generator_loss` = `cross_entropy_loss` + `alpha` * `mse_loss`
            The higher the `alpha`, the more the mse_loss will affect the
            overall generator loss.

        The following parameters are optional to allow
        for advanced use case when you want to pass
        a customized Generator or Discriminator model.

        generator: A customized Generator model that can fit right in
            with the architecture.
            If specified, it will replace the default generator instance
            created by the model.
            This allows you to take full control over the Generator architecture.
            Note that, you import the standalone `Generator` model
            `from teras.models.gain import Generator` customize it through
            available params, subclass it or construct your own Generator
            from scratch given that it can fit within the architecture,
            for instance, satisfy the input/output requirements.
        discriminator: A customized Discriminator model that can fit right in
            with the architecture.
            Everything specified about generator above applies here as well.

    Example:
        ```python
        input_data = tf.random.uniform([200, 10])

        from teras.utils.gain import inject_missing_values
        input_data = inject_missing_values(input_data)

        # GAIN requires data in a specific format for which we have
        # relevant DataSampler and DataTrasnformer classes
        from teras.preprocessing.gain import DataSampler, DataTransformer

        # Assume we have following numerical and categorical features
        numerical_features = ["num_1", "num_2"]
        categorical_features = ["cat_1", "cat_2", "cat_3"]

        data_transformer = DataTransformer(numerical_features=numerical_features,
                                           categorical_features=categorical_features)

        transformed_data = data_transformer.transform(input_data, return_dataframe=True)

        data_sampler = DataSampler()
        dataset = data_sampler.get_dataset(transformed_data)

        # Instantiate GAIN
        gain_imputer = GAIN()

        # Compile it
        gain_imputer.compile()

        # Train it
        gain_imputer.fit(dataset)

        # Predict
        test_data = transformed_data[:50]
        imputed_data = gain_imputer.predict(test_data)

        # Reverse transform into original format
        imputed_data = data_sampler.reverse_transform(imputed_data)
        ```
    """
    def __init__(self,
                 hint_rate: float = 0.9,
                 alpha: float = 100,
                 generator: keras.Model = None,
                 discriminator: keras.Model = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.generator = generator
        self.discriminator = discriminator

        if self.generator is None:
            self.generator = Generator()
        if self.discriminator is None:
            self.discriminator = Discriminator()

        self.z_sampler = tfp.distributions.Uniform(low=0.,
                                                   high=0.01,
                                                   name="z_sampler")
        self.hint_vectors_sampler = tfp.distributions.Binomial(total_count=1,
                                                               probs=self.hint_rate,
                                                               name="hint_vectors_generator")

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def compile(self,
                generator_optimizer=optimizers.Adam(),
                discriminator_optimizer=optimizers.Adam(),
                generator_loss=generator_loss,
                discriminator_loss=discriminator_loss):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def call(self, inputs, mask=None, training=None):
        if mask is not None:
            inputs = tf.concat([inputs, mask], axis=1)
        generated_samples = self.generator(inputs)
        return generated_samples

    def train_step(self, data):
        # data is a tuple of x_generator and x_discriminator batches
        # drawn from the dataset. The reason behind generating two separate
        # batches of data at each step is that it's how GAIN's algorithm works
        x_gen, x_disc = data

        # =====> Train the discriminator <=====
        # Create mask
        mask = tf.constant(1.) - tf.cast(tf.math.is_nan(x_disc), dtype=tf.float32)
        # replace nans with 0.
        x_disc = tf.where(tf.math.is_nan(x_disc), x=0., y=x_disc)
        # Sample noise
        z = self.z_sampler.sample(sample_shape=tf.shape(x_disc))
        # Sample hint vectors
        hint_vectors = self.hint_vectors_sampler.sample(sample_shape=(tf.shape(x_disc)))
        hint_vectors *= mask
        # Combine random vectors with original data
        x_disc = x_disc * mask + (1 - mask) * z
        # Keras model raises `gain`'s weights not created error if we don't implement the
        # `call` method and call it from our code no matter if we compile it or not.
        # so to work around that error we place the call to `Generator`'s `call` method
        # in the `call` method of the `GAIN` model, this way the model's
        # generated_samples = self.generator(tf.concat([x_disc, mask], axis=1))
        generated_samples = self(x_disc, mask=mask)
        # Combine generated samples with original data
        x_hat_disc = (generated_samples * (1 - mask)) + (x_disc * mask)
        with tf.GradientTape() as tape:
            discriminator_pred = self.discriminator(tf.concat([x_hat_disc, hint_vectors], axis=1))
            loss_disc = self.discriminator_loss(discriminator_pred, mask)
        gradients = tape.gradient(loss_disc, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        # =====> Train the generator <=====
        mask = tf.constant(1.) - tf.cast(tf.math.is_nan(x_gen), dtype=tf.float32)
        x_gen = tf.where(tf.math.is_nan(x_gen), x=0., y=x_gen)
        z = self.z_sampler.sample(sample_shape=tf.shape(x_gen))
        hint_vectors = self.hint_vectors_sampler.sample(sample_shape=(tf.shape(x_gen)))
        hint_vectors *= mask
        x_gen = x_gen * mask + (1 - mask) * z

        with tf.GradientTape() as tape:
            # generated_samples = self.generator(tf.concat([x_gen, mask], axis=1))
            generated_samples = self(x_gen, mask=mask)
            # Combine generated samples with original/observed data
            x_hat = (generated_samples * (1 - mask)) + (x_gen * mask)
            discriminator_pred = self.discriminator(tf.concat([x_hat, hint_vectors], axis=1))
            loss_gen = self.generator_loss(generated_samples=generated_samples,
                                           real_samples=x_gen,
                                           discriminator_pred=y_preds,
                                           mask=mask,
                                           alpha=self.alpha)
        gradients = tape.gradient(loss_gen, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        # Update custom tracking metrics
        self.generator_loss_tracker.update_state(loss_gen)
        self.discriminator_loss_tracker.update_state(loss_disc)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"generator_" + m.name: m.result() for m in self.generator.metrics})
        results.update({"discriminator_" + m.name: m.result() for m in self.discriminator.metrics})
        return results

    def predict_step(self, data):
        """
        Args:
            Transformed data.
        Returns:
            Imputed data that should be reverse transformed
            to its original form.
        """
        if isinstance(data, tuple):
            data = data[0]
        data = tf.cast(data, tf.float32)
        # Create mask
        mask = tf.constant(1.) - tf.cast(tf.math.is_nan(data), dtype=tf.float32)
        data = tf.where(tf.math.is_nan(data), x=0., y=data)
        # Sample noise
        z = self.z_sampler.sample(sample_shape=tf.shape(data))
        x = mask * data + (1 - mask) * z
        # imputed_data = self.generator(tf.concat([x, mask], axis=1))
        imputed_data = self(x, mask=mask)
        imputed_data = mask * data + (1 - mask) * imputed_data
        return imputed_data

    def get_config(self):
        config = super(GAIN, self).get_config()
        config.update({'hint_rate': self.hint_rate,
                       'alpha': self.alpha,
                       'generator': self.generator,
                       'discriminator': self.discriminator})
        return config
