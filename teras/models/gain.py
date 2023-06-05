import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from typing import List, Tuple, Union


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class Generator(keras.Model):
    """
    Generator based on the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        hidden_dims: A list of hidden dimensions.
            For each element, a new dense layer will be added.
            In official implementation, `hidden_dim` is equal to the input_dim,
            and the number of hidden layers is 2.
            So by default we pass [input_dim, input_dim].
        activation_hidden: Activation function to use for hidden layers.
            Defaults to 'relu'.
        kernel_initializer: Initializer for weights of layers within generator.
            Defaults to 'glorot_normal' a.k.a. 'xavier normal'.
        activation_hidden: Activation function to use for the output layer.
            Defaults to 'sigmoid'.
    """
    def __init__(self,
                 hidden_dims=None,
                 activation_hidden="relu",
                 kernel_initializer="glorot_normal",
                 activation_out="sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.activation_hidden = activation_hidden
        self.kernel_initializer = kernel_initializer
        self.activation_out = activation_out

        self.hidden_block = models.Sequential(name="generator_hidden_block")

    def build(self, input_shape):
        # inputs is the concatenation of `mask` and `original data`
        # and `mask` has the same dimensions as `data`, so we halve the
        # `input_shape[1]` here to get the `original data` dimensions
        data_dim = input_shape[1] // 2
        if self.hidden_dims is None:
            self.hidden_dims = [data_dim] * 2
        for dim in self.hidden_dims:
            self.hidden_block.add(layers.Dense(dim,
                                                     activation=self.activation_hidden,
                                                     kernel_initializer=self.kernel_initializer))
        self.dense_out = layers.Dense(data_dim,
                                            activation=self.activation_out,
                                            kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        # inputs is the concatenation of mask and original data
        h = self.hidden_block(inputs)
        probs = self.dense_out(h)
        return probs

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({'hidden_dims': self.hidden_dims,
                       'activation_hidden': self.activation_hidden,
                       'activation_out': self.activation_out,
                       'kernel_initializer': self.kernel_initializer})
        return config


class Discriminator(keras.Model):
    """
    Discriminator based on the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        hidden_dims: A list of hidden dimensions.
            For each element, a new dense layer will be added.
            In official implementation, `hidden_dim` is equal to the input_dim,
            and the number of hidden layers is 2.
            So by default we pass [input_dim, input_dim].
        activation_hidden: Activation function to use for hidden layers.
            Defaults to 'relu'.
        kernel_initializer: Initializer for weights of layers within generator.
            Defaults to 'glorot_normal' a.k.a. 'xavier normal'.
        activation_hidden: Activation function to use for the output layer.
            Defaults to 'sigmoid'.
    """
    def __init__(self,
                 hidden_dims=None,
                 activation_hidden="relu",
                 kernel_initializer="glorot_normal",
                 activation_out="sigmoid",
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.activation_hidden = activation_hidden
        self.kernel_initializer = kernel_initializer
        self.activation_out = activation_out

        self.hidden_block = models.Sequential(name="discriminator_hidden_block")

    def build(self, input_shape):
        # inputs is the concatenation of `hint` and original data
        # and `hint` has the same dimensions as data, so we halve the
        # input_shape[1] here to get the original data dimensions
        data_dim = input_shape[1] // 2
        if self.hidden_dims is None:
            self.hidden_dims = [data_dim] * 2
        for dim in self.hidden_dims:
            self.hidden_block.add(layers.Dense(dim,
                                                     activation=self.activation_hidden,
                                                     kernel_initializer=self.kernel_initializer))
        self.dense_out = layers.Dense(data_dim,
                                            activation=self.activation_out,
                                            kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        # inputs is the concatenation of `hint` and `original data`
        outputs = self.hidden_block(inputs)
        outputs = self.dense_out(outputs)
        return outputs

    def get_config(self):
        config = super(Discriminator, self).get_config()
        config.update({'hidden_dims': self.hidden_dims,
                       'activation_hidden': self.activation_hidden,
                       'activation_out': self.activation_out,
                       'kernel_initializer': self.kernel_initializer})
        return config


class GAIN(keras.Model):
    """
    GAIN model based on the architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        hint_rate: Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Should be between 0. and 1.
            Hint vectors ensure that generated samples follow the
            underlying data distribution.
        alpha: Hyper parameter for the generator loss computation that
            controls how much weight should be given to the MSE loss.
            Precisely, `generator_loss` = `cross_entropy_loss` + `alpha` * `mse_loss`
            The higher the `alpha`, the more the mse_loss will affect the
            overall generator loss.
        generator_hidden_dims: A list of hidden dimensions for generator.
            For each element, a new dense layer will be added.
            In official implementation, `hidden_dim` for generator
            is equal to the input_dim, and the number of hidden layers is 2.
            So by default we pass [input_dim, input_dim] to generator.
        discriminator_hidden_dims: A list of hidden dimensions for discriminator.
            For each element, a new dense layer will be added.
            In official implementation, `hidden_dim` for discriminator
            is equal to the input_dim, and the number of hidden layers is 2.
            So by default we pass [input_dim, input_dim] to discriminator.
    """
    def __init__(self,
                 hint_rate: float = 0.9,
                 alpha: float = 100,
                 generator_hidden_dims: LIST_OR_TUPLE = None,
                 discriminator_hidden_dims: LIST_OR_TUPLE = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.generator_hidden_dims = generator_hidden_dims
        self.discriminator_hidden_dims = discriminator_hidden_dims

        self.generator = Generator(hidden_dims=self.generator_hidden_dims)
        self.discriminator = Discriminator(hidden_dims=self.discriminator_hidden_dims)

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
                gen_optimizer=None,
                disc_optimizer=None,
                gen_loss=None,
                disc_loss=None):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss

    def get_generator(self):
        return self.generator

    def call(self, inputs, mask=None, training=None):
        if mask is not None:
            inputs = tf.concat([inputs, mask], axis=1)
        gen_outs = self.generator(inputs)
        return gen_outs

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
            y_preds = self.discriminator(tf.concat([x_hat_disc, hint_vectors], axis=1))
            loss_disc = self.disc_loss(y_preds, mask)
        gradients = tape.gradient(loss_disc, self.discriminator.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

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
            y_preds = self.discriminator(tf.concat([x_hat, hint_vectors], axis=1))
            loss_gen = self.gen_loss(generated_samples=generated_samples,
                                           real_samples=x_gen,
                                           y_preds=y_preds,
                                           mask=mask,
                                           alpha=self.alpha)
        gradients = tape.gradient(loss_gen, self.generator.trainable_weights)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

        # Update custom tracking metrics
        self.generator_loss_tracker.update_state(loss_gen)
        self.discriminator_loss_tracker.update_state(loss_disc)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"gen_" + m.name: m.result() for m in self.generator.metrics})
        results.update({"disc_" + m.name: m.result() for m in self.discriminator.metrics})
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
                       'generator_hidden_dims': self.generator_hidden_dims,
                       'discriminator_hidden_dims': self.discriminator_hidden_dims})
        config.update({'generator': self.generator.get_config(),
                       'discriminator': self.discriminator.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)