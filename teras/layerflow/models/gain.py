import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from teras.losses.gain import generator_loss, discriminator_loss


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class GAIN(keras.Model):
    """
    GAIN model with LayerFlow design.
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
        generator: ``keras.Model``,
            An instance of ``GAINGenerator`` model or any customized model that can
            work in its place.
            This allows you to take full control over the Generator's architecture.
            You import the standalone ``GAINGenerator`` model as follows,
                >>> from teras.models import GAINGenerator

        discriminator: ``keras.Model``,
            An instance of ``GAINDiscriminator`` model or any customized model that
            can work in its place.
            This allows you to take full control over the Discriminator's architecture.
            You import the standalone ``GAINDiscriminator`` model as follows,
                >>> from teras.models import GAINDiscriminator

        num_discriminator_steps: ``int``, default 1,
            Number of discriminator training steps per GAIN training step.

        hint_rate: ``float``, default 0.9,
            Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Must be between 0. and 1.
            Hint vectors ensure that generated samples follow the
            underlying data distribution.

        alpha: ``float``, default 100,
            Hyper parameter for the generator loss computation that
            controls how much weight should be given to the MSE loss.
            Precisely, `generator_loss` = `cross_entropy_loss` + `alpha` * `mse_loss`
            The higher the `alpha`, the more the mse_loss will affect the
            overall generator loss.

    Example:
        ```python
        input_data = pd.DataFrame(tf.random.uniform([200, 5]),
                                columns=["A", "B", "C", "D", "E"])

        from teras.utils.gain import inject_missing_values
        input_data = inject_missing_values(input_data)

        # GAIN requires data in a specific format for which we have
        # relevant DataSampler and DataTrasnformer classes
        from teras.preprocessing.gain import DataSampler, DataTransformer

        numerical_features = ["A", "B"]
        categorical_features = ["C", "D", "E"]

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

        # Imputation Method 1:
        imputed_data = gain_imputer.predict(test_data)

        # Reverse transform into original format
        imputed_data = data_transformer.reverse_transform(imputed_data)

        # Imputation Method 2:
        imputed_data = gain_imputer.impute(test_data)
        ```
    """
    def __init__(self,
                 generator: keras.Model,
                 discriminator: keras.Model,
                 num_discriminator_steps: int = 1,
                 hint_rate: float = 0.9,
                 alpha: float = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.num_discriminator_steps = num_discriminator_steps
        self.hint_rate = hint_rate
        self.alpha = alpha

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
                generator_optimizer=keras.optimizers.Adam(),
                discriminator_optimizer=keras.optimizers.Adam(),
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
        for _ in range(self.num_discriminator_steps):
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
                                           discriminator_pred=discriminator_pred,
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

    def impute(self,
               x,
               data_transformer=None,
               reverse_transform=True,
               batch_size=None,
               verbose="auto",
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False
               ):
        """
        Impute function combines GAIN's `predict` method and
        DataTransformer's `reverse_transform` method to fill
        the missing data and transform into the original format.
        It exposes all the arguments taken by the `predict` method.

        Args:
            x: `pd.DataFrame`, dataset with missing values.
            data_transformer: An instance of DataTransformer class which
                was used in transforming the raw input data
            reverse_transform: `bool`, default True, whether to reverse
                transformed the raw imputed data to original format.

        Returns:
            Imputed data in the original format.
        """
        x_imputed = self.predict(x,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 steps=steps,
                                 callbacks=callbacks,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)
        if reverse_transform:
            if data_transformer is None:
                raise ValueError("To reverse transform the raw imputed data, `data_transformer` must not be None. "
                                 "Please pass the instance of DataTransformer class used to transform the input "
                                 "data as argument to this `impute` method."
                                 "Or alternatively, you can set `reverse_transform` parameter to False, "
                                 "and manually reverse transform the generated raw data to original format "
                                 "using the `reverse_transform` method of DataTransformer instance.")
            x_imputed = data_transformer.reverse_transform(x_imputed)
        return x_imputed

    def get_config(self):
        config = super().get_config()
        new_config = {'generator': keras.layers.serialize(self.generator),
                      'discriminator': keras.layers.serialize(self.discriminator)
                      }
        config.update(new_config)
        return config
