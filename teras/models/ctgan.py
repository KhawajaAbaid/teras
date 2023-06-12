import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from teras.layers.ctgan import GeneratorBlock, DiscriminatorBlock
from teras.layers.activations import GumbelSoftmax
from teras.losses.ctgan import generator_loss, discriminator_loss
from teras.preprocessing.ctgan import DataTransformer, DataSampler
from typing import List, Union, Tuple
from tqdm import tqdm
from warnings import warn


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
HIDDEN_BLOCK_TYPE = Union[keras.layers.Layer, keras.models.Model]


class Generator(keras.Model):
    """
    Generator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: `int`, dimensionality of the dataset.
            It will also be the dimensionality of the output produced
            by the generator.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
        units_values: default [256, 256], A list or tuple of units.
            For each value, a `GeneratorBlock`
            (`from teras.layers.ctgan import GeneratorBlock`)
            of that dimensionality (units) is added to the generator
            to form the `hidden block` of the generator.
        hidden_block: `layers.Layer` or `keras.Model`. If you want more control
            over the hidden block than simply altering the units values,
            you can create your own hidden block and pass it as argument.
            In this case, you have full control of the hidden block.
            Note that if you specify a hidden block, the `units_values` parameter
            will be ignored.
            If `None`, a hidden block is constructed with `GeneratorBlock`
            (`from teras.layers.ctgan import GeneratorBlock`)
            where the number and dimensionality of blocks is determined by the
            `units_values` argument.
        output_layer: `layers.Layer`. If you want full control over the
            output_layer you can create your own custom layer and
            pass it as argument.
            By default, a simple `Dense` layer of `data_dim` dimensionality
            with no activation is used as the output layer.
        meta_data: `namedtuple`. The Generator in CTGAN architecture,
            applies different activation functions to the output of Generator,
            depending on the type of features.
            And to determine the feature types and for other computation during
            activation step, the `meta data` computed during the data transformation step,
            is required.
            It can be accessed through the `.get_meta_data()` method of the DataTransformer
            instance which was used to transform the raw input data.
            It must NOT be None.
    """
    def __init__(self,
                 data_dim: int,
                 units_values: LIST_OR_TUPLE = (256, 256),
                 hidden_block: HIDDEN_BLOCK_TYPE = None,
                 output_layer: keras.layers.Layer = None,
                 meta_data=None,
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Generator residual blocks and the dimensionality of those blocks.
                        But {units_values} was passed.""")

        if hidden_block is not None and units_values is not None:
            warn(f"A custom hidden block was specified, the `units_values` {units_values} "
                 f"will be ignored.")

        if meta_data is None:
            raise ValueError(f"""`meta_data`, which is computed during the data transformation step,
                    is required by the Generator to apply relevant activation functions to the 
                    output of the Generator. But {meta_data} was passed.
                    Please pass the meta data by accessing it through the `.get_meta_data()` method
                    of the DataTransformer instance which was used to transform the raw input data.
                    """)

        self.data_dim = data_dim
        self.units_values = units_values
        self.hidden_block = hidden_block
        self.output_layer = output_layer
        self.meta_data = meta_data

        if self.hidden_block is None:
            self.hidden_block = models.Sequential(name="generator_hidden_block")
            for units in self.units_values:
                self.hidden_block.add(GeneratorBlock(units))

        if self.output_layer is None:
            self.output_layer = layers.Dense(self.data_dim, name="generator_output_layer")

        self.gumbel_softmax = GumbelSoftmax()

    def apply_activations_by_feature_type(self, interim_outputs):
        """
        This function applies activation functions to the interim outputs of
        the Generator by feature type.
        As CTGAN architecture requires specific transformations on the raw input data,
        that decompose one feature in several features,
        and since each type of feature, i.e. numerical or categorical require
        different activation functions to be applied, the process of applying those
        activations becomes rather tricky as it requires knowledge of underlying
        data transformation and features meta data.
        To ease the user's burden, in case a user wants to subclass this
        Generator model and completely customize the inner workings of the generator
        but would want to use the activation method specific to the CTGAN architecture,
        so that the subclassed Generator can work seamlessly with the rest of the
        architecture and there won't be any discrepancies in outputs produced by
        the subclasses Generator and those expected by the architecture,
        this function is separated, so user can just call this function on the interim
        outputs in the `call` method.

        Args:
            interim_outputs: Outputs produced by the `output_layer` of the Generator.

        Returns:
            Final outputs activated by the relevant activation functions.
        """
        outputs = []
        numerical_features_relative_indices = self.meta_data.numerical.relative_indices_all
        features_relative_indices_all = self.meta_data.relative_indices_all
        num_valid_clusters_all = self.meta_data.numerical.num_valid_clusters_all
        cont_i = 0
        cat_i = 0
        num_categories_all = self.meta_data.categorical.num_categories_all
        for i, index in enumerate(features_relative_indices_all):
            # the first k = num_numerical_features are numerical in the data
            if i < len(numerical_features_relative_indices):
                # each numerical features has been transformed into num_valid_clusters + 1 features
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

    def call(self, inputs):
        # inputs have the shape |z| + |cond|
        # while the outputs will have the shape of equal to (batch_size, transformed_data_dims)
        interim_outputs = self.output_layer(self.hidden_block(inputs))
        outputs = self.apply_activations_by_feature_type(interim_outputs)
        return outputs


class Discriminator(keras.Model):
    """
    Discriminator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        units_values: default [256, 256], A list or tuple of units.
            For each value, a `DiscriminatorBlock`
            (`from teras.layers.ctgan import DiscriminatorBlock`)
            of that dimensionality (units) is added to the discriminator
            to form the `hidden block` of the discriminator.
        hidden_block: `layers.Layer` or `keras.Model`. If you want more control
            over the hidden block than simply altering the units values,
            you can create your own hidden block and pass it as argument.
            In this case, you have full control of the hidden block.
            Note that if you specify a hidden block, the `units_values` parameter
            will be ignored.
            If `None`, a hidden block is constructed with `DiscriminatorBlock`
            (`from teras.layers.ctgan import DiscriminatorBlock`)
            where the number and dimensionality of blocks is determined by the
            `units_values` argument.
        output_layer: `layers.Layer`. If you want full control over the
            output_layer you can create your own custom layer and
            pass it as argument.
            By default, a simple `Dense` layer of `1` dimensionality
            with no activation is used as the output layer.
        packing_degree: `int`, default 8, Packing degree - taken from the PacGAN paper.
            The number of samples concatenated or "packed" together.
            It must be a factor of the batch_size.
            Packing degree is borrowed from the PacGAN [Diederik P. Kingma et al.] architecture,
            which proposes passing `m` samples at once to discriminator instead of `1` to be
            jointly classified as real or fake by the discriminator in order to tackle the
            issue of mode collapse inherent in the GAN based architectures.
            The number of samples passed jointly `m`, is termed as the `packing degree`.
        gradient_penalty_lambda: default 10, Controls the strength of gradient penalty.
                lambda value is directly proportional to the strength of gradient penalty.
                Gradient penalty penalizes the discriminator for large weights in an attempt
                to combat Discriminator becoming too confident and overfitting.

    Example:
        ```python
        # Instantiate Generator
        generator = Generator(data_dim=data_dim,
                              meta_data=meta_data)

        # Instantiate Discriminator
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
                 units_values: LIST_OR_TUPLE = (256, 256),
                 hidden_block: HIDDEN_BLOCK_TYPE = None,
                 output_layer: keras.layers.Layer = None,
                 packing_degree: int = 8,
                 gradient_penalty_lambda=10,
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Discriminator blocks and the dimensionality of those blocks.
                        But {units_values} was passed""")

        if hidden_block is not None and units_values is not None:
            warn(f"A custom hidden block was specified, the `units_values` {units_values} "
                 f"will be ignored.")

        self.units_values = units_values
        self.hidden_block = hidden_block
        self.output_layer = output_layer
        self.packing_degree = packing_degree
        self.gradient_penalty_lambda = gradient_penalty_lambda

        if self.hidden_block is None:
            self.hidden_block = models.Sequential(name="discriminator_hidden_block")
            for units in self.units_values:
                self.hidden_block.add(DiscriminatorBlock(units))

        if self.output_layer is None:
            self.output_layer = layers.Dense(1, name="discriminator_output_layer")

    @tf.function
    def gradient_penalty(self,
                         real_samples,
                         generated_samples):
        """
        Calculates the gradient penalty as proposed
        in the paper "Improved Training of Wasserstein GANs"

        Reference(s):
            https://arxiv.org/abs/1704.00028

        Args:
            real_samples: Data samples drawn from the real dataset
            generated_samples: Data samples generated by the generator

        Returns:
            Gradient penalty computed for given values.
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
        gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0)) * self.gradient_penalty_lambda
        return gradient_penalty

    def call(self, inputs):
        inputs_dim = tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, shape=(-1, self.packing_degree * inputs_dim))
        outputs = self.hidden_block(inputs)
        outputs = self.output_layer(outputs)
        return outputs


class CTGAN(keras.Model):
    """
    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

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
        latent_dim: `int`, default 128, Dimensionality of noise or `z` that serves as
            input to Generator to generate samples.
        packing_degree: `int`, default 8, Packing degree - taken from the PacGAN paper.
            The number of samples concatenated or "packed" together.
            It must be a factor of the batch_size.
            Packing degree is borrowed from the PacGAN [Diederik P. Kingma et al.] architecture,
            which proposes passing `m` samples at once to discriminator instead of `1` to be
            jointly classified as real or fake by the discriminator in order to tackle the
            issue of mode collapse inherent in the GAN based architectures.
            The number of samples passed jointly `m`, is termed as the `packing degree`.
        meta_data: `namedtuple`. Simply pass the result of `.get_meta_data()`
            method of the DataTransformer instance which was used to transform the raw input data.
            The Generator in CTGAN architecture applies different activation functions
            to the output of Generator, depending on the type of features.
            And to determine the feature types and for other computation during
            activation step, the `meta data` computed during the data transformation step,
            is required.
            It is also required during the computation of generator loss.
            Hence, it cannot be None.
            It can be accessed through the `.get_meta_data()` method of the DataTransformer
            instance which was used to transform the raw input data.
    """
    def __init__(self,
                 generator: keras.Model = None,
                 discriminator: keras.Model = None,
                 num_discriminator_steps: int = 1,
                 data_dim: int = None,
                 latent_dim: int = 128,
                 packing_degree=8,
                 meta_data=None,
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
        self.latent_dim = latent_dim
        self.packing_degree = packing_degree
        self.meta_data = meta_data

        # If user specifies a custom generator, we won't instantiate Generator.
        if self.generator is None:
            # Instantiate Generator
            self.generator = Generator(data_dim=self.data_dim,
                                       meta_data=self.meta_data)

        # If user specifies a custom discriminator, we won't instantiate Discriminator.
        if discriminator is None:
            # Instantiate discriminator
            self.discriminator = Discriminator()

        # Loss trackers
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def compile(self,
                generator_optimizer=optimizers.Adam(learning_rate=1e-3,
                                                    beta_1=0.5, beta_2=0.9),
                discriminator_optimizer=optimizers.Adam(learning_rate=1e-3,
                                                        beta_1=0.5, beta_2=0.9),
                generator_loss=generator_loss,
                discriminator_loss=discriminator_loss,
                ):
        super().compile()
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
        # generator_partial_loss_fn = partial(generator_loss, mask=mask, meta_data=self.meta_data)
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
                                           meta_data=self.meta_data)
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
                 reverse_transform: bool = True,
                 batch_size: int = None):
        """
        Generates new samples using the trained generator.

        Args:
            num_samples: Number of new samples to generate
            data_sampler: Instance of the DataSampler class used in preparing
                the tensorflow dataset for training.
            data_transformer: Instance of DataTransformer class used to preprocess
                the raw data.
                This is required only if the `reverse_transform` is set to True.
            reverse_transform: bool, default True,
                whether to reverse transform the generated data to the original data format.
                If False, the raw generated data will be returned, which you can then manually
                transform into original data format by utilizing DataTransformer instance's
                `reverse_transform` method.
            batch_size: int, default None.
                If a value is passed, samples will be generated in batches
                where `batch_size` determines the size of each batch.
                If `None`, all `num_samples` will be generated at once.
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
                raise ValueError("""To reverse transform the raw generated data, `data_transformer` must not be None.
                             Please pass the instance of DataTransformer class used to transform the input
                             data. Or alternatively, you can set `reverse_transform` to False, and later
                             manually reverse transform the generated raw data to original format.""")
            generated_samples = data_transformer.reverse_transform(x_generated=generated_samples)

        return generated_samples
