import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from teras.losses.gain import generator_loss, discriminator_loss
from teras.preprocessing.gain import DataTransformer, DataSampler
from teras.layers.gain import GeneratorBlock, DiscriminatorBlock
from typing import List, Tuple, Union
from warnings import warn


LIST_OR_TUPLE = Union[List[int], Tuple[int]]


class Generator(keras.Model):
    """
    Generator model for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        data_dim: `int`, dimensionality of the dataset.
            It will also be the dimensionality of the output produced
            by the generator.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the `fit` method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
        units_values: default `None`, A list or tuple of units.
            For each value, a `GeneratorBlock`
            (`from teras.layers.gain import GeneratorBlock`)
            of that dimensionality (units) is added to the generator
            to form the `hidden block` of the generator.
        hidden_block: `layers.Layer` or `keras.Model`. If you want more control
            over the hidden block than simply altering the units values,
            you can create your own hidden block and pass it as argument.
            In this case, you have full control of the hidden block.
            Note that if you specify a hidden block, the `units_values` parameter
            will be ignored.
            If `None`, a hidden block is constructed with `GeneratorBlock`
            (`from teras.layers.gain import GeneratorBlock`)
            where the number and dimensionality of blocks is determined by the
            `units_values` argument.
        output_layer: `layers.Layer`. If you want full control over the
            output_layer you can create your own custom layer and
            pass it as argument.
            By default, a simple `Dense` layer of `data_dim` dimensionality
            wit "sigmoid" activation is used as the output layer.

    Example:
        ```python
        data_dim = 12
        # Instantiate Generator.
        generator = Generator(data_dim=data_dim)

        # Sample noise to generate samples from
        z = tf.random.normal([512, 18])

        # Generate samples
        generated_samples = generator(z)
        ```
    """
    def __init__(self,
                 data_dim: int,
                 units_values: LIST_OR_TUPLE = None,
                 hidden_block: keras.layers.Layer = None,
                 output_layer: keras.layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)

        if units_values is not None and not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Generator blocks and the dimensionality of those blocks.
                        But {units_values} was passed.""")

        if hidden_block is not None and units_values is not None:
            warn(f"A custom hidden block was specified, the `units_values` {units_values} "
                 f"will be ignored.")

        self.data_dim = data_dim
        self.units_values = units_values
        self.hidden_block = hidden_block
        self.output_layer = output_layer

        if self.hidden_block is None:
            if self.units_values is None:
                self.units_values = [self.data_dim] * 2
            self.hidden_block = keras.models.Sequential(name="generator_hidden_block")
            for units in self.units_values:
                self.hidden_block.add(GeneratorBlock(units))

        if self.output_layer is None:
            self.output_layer = layers.Dense(self.data_dim,
                                             activation="sigmoid")

    def call(self, inputs):
        # inputs is the concatenation of `mask` and `original data`
        # where `mask` has the same dimensions as data
        # so the inputs received are 2x the dimensions of `original data`
        outputs = self.hidden_block(inputs)
        outputs = self.output_layer(outputs)
        return outputs


class Discriminator(keras.Model):
    """
    Discriminator model for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Note that the Generator and Discriminator share the exact same
    architecture by default. They differ in the inputs they receive
    and their loss functions.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        data_dim: `int`, dimensionality of the dataset.
            It will also be the dimensionality of the output produced
            by the generator.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the `fit` method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
        units_values: default `None`, A list or tuple of units.
            For each value, a `DiscriminatorBlock`
            (`from teras.layers.gain import DiscriminatorBlock`)
            of that dimensionality (units) is added to the generator
            to form the `hidden block` of the generator.
        hidden_block: `layers.Layer` or `keras.Model`. If you want more control
            over the hidden block than simply altering the units values,
            you can create your own hidden block and pass it as argument.
            In this case, you have full control of the hidden block.
            Note that if you specify a hidden block, the `units_values` parameter
            will be ignored.
            If `None`, a hidden block is constructed with `DiscriminatorBlock`
            (`from teras.layers.gain import DiscriminatorBlock`)
            where the number and dimensionality of blocks is determined by the
            `units_values` argument.
        output_layer: `layers.Layer`. If you want full control over the
            output_layer you can create your own custom layer and
            pass it as argument.
            By default, a simple `Dense` layer of `data_dim` dimensionality
            wit "sigmoid" activation is used as the output layer.

    Example:
        ```python
        data_dim = 12
        # Instantiate Generator
        generator = Generator(data_dim)

        # Instantiate Discriminator
        discriminator = Discriminator(data_dim)

        # Sample noise to generate samples from
        z = tf.random.normal([512, 18])

        # Generate samples
        generated_samples = generator(z)

        # Predict using discriminator
        y_pred = discriminator(generated_samples)
        ```
    """
    def __init__(self,
                 data_dim: int,
                 units_values: LIST_OR_TUPLE = None,
                 hidden_block: keras.layers.Layer = None,
                 output_layer: keras.layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)

        if units_values is not None and not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Discriminator blocks and the dimensionality of those blocks.
                        But {units_values} was passed.""")

        if hidden_block is not None and units_values is not None:
            warn(f"A custom hidden block was specified, the `units_values` {units_values} "
                 f"will be ignored.")

        self.data_dim = data_dim
        self.units_values = units_values
        self.hidden_block = hidden_block
        self.output_layer = output_layer

        if self.hidden_block is None:
            if self.units_values is None:
                self.units_values = [self.data_dim] * 2
            self.hidden_block = keras.models.Sequential(name="discriminator_hidden_block")
            for units in self.units_values:
                self.hidden_block.add(DiscriminatorBlock(units))

        if self.output_layer is None:
            self.output_layer = layers.Dense(self.data_dim,
                                             activation="sigmoid")

    def call(self, inputs):
        # inputs is the concatenation of `hint` and manipulated
        # Generator output (i.e. generated samples).
        # `hint` has the same dimensions as data
        # so the inputs received are 2x the dimensions of original data
        outputs = self.hidden_block(inputs)
        outputs = self.output_layer(outputs)
        return outputs


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
        generator: `keras.Model`, a customized Generator model that can
            fit right in with the architecture.
            If specified, it will replace the default generator instance
            created by the model.
            This allows you to take full control over the Generator architecture.
            Note that, you import the standalone `Generator` model
            `from teras.models.gain import Generator` customize it through
            available params, subclass it or construct your own Generator
            from scratch given that it can fit within the architecture,
            for instance, satisfy the input/output requirements.
        discriminator: `keras.Model`, a customized Discriminator model that
            can fit right in with the architecture.
            Everything specified about generator above applies here as well.
        num_discriminator_steps: `int`, default 1, Number of discriminator training steps
            per CTGAN training step.
        data_dim: `int`, dimensionality of the input dataset.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
            This parameter can be left None if instances of Generator and Discriminator
            are passed, otherwise it must be specified.
        hint_rate: Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Must be between 0. and 1.
            Hint vectors ensure that generated samples follow the
            underlying data distribution.
        alpha: Hyper parameter for the generator loss computation that
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
                 generator: keras.Model = None,
                 discriminator: keras.Model = None,
                 num_discriminator_steps: int = 1,
                 data_dim: int = None,
                 hint_rate: float = 0.9,
                 alpha: float = 100,
                 **kwargs):
        super().__init__(**kwargs)

        if data_dim is None and (generator is None and discriminator is None):
            raise ValueError(f"""`data_dim` is required to instantiate the Generator and Discriminator objects.
                    But {data_dim} was passed.
                    You can either pass the value for `data_dim` -- which can be accessed through `.data_dim`
                    attribute of DataSampler instance if you don't know the data dimensions --
                    or you can instantiate and pass your own Generator and Discriminator instances,
                    in which case you can leave the `data_dim` parameter as None.""")

        self.generator = generator
        self.discriminator = discriminator
        self.num_discriminator_steps = num_discriminator_steps
        self.data_dim = data_dim
        self.hint_rate = hint_rate
        self.alpha = alpha

        if self.generator is None:
            self.generator = Generator(data_dim=self.data_dim)

        if self.discriminator is None:
            self.discriminator = Discriminator(data_dim=self.data_dim)

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
        config = super(GAIN, self).get_config()
        config.update({'generator': self.generator,
                       'discriminator': self.discriminator,
                       'num_discriminator_steps': self.num_discriminator_steps,
                       'data_dim': self.data_dim,
                       'hint_rate': self.hint_rate,
                       'alpha': self.alpha,
                       })
        return config
