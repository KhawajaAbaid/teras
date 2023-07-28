import tensorflow as tf
from tensorflow import keras
from teras.layers.ctgan.ctgan_generator_block import CTGANGeneratorBlock
from teras.layers.ctgan.ctgan_discriminator_block import CTGANDiscriminatorBlock
from teras.layerflow.models.ctgan import CTGAN as _CTGAN_LF
from teras.layers.activation import GumbelSoftmax
from teras.utils.types import UnitsValuesType


@keras.saving.register_keras_serializable(package="teras.models")
class CTGANGenerator(keras.Model):
    """
    CTGANGenerator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        data_dim: ``int``,
            The dimensionality of the dataset.
            It will also be the dimensionality of the output produced
            by the generator.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionailty of the transformed dataset throught the
            ``.data_dim`` attribute of the ``CTGANDataSampler`` instance used in sampling
            the dataset.

        metadata: ``dict``,
            The ``CTGANGeneratorBlock``, applies different activation functions
            to its outputs depending on the type of features (categorical or numerical).
            And to determine the feature types and for other computations during the
            activation step, the ``metadata`` computed during the data transformation step,
            is required.

            It can be accessed through the ``.get_metadata()`` method of the ``CTGANDataTransformer``
            instance which was used to transform the raw input data.

            Note that, this is NOT the same metadata as ``features_metadata``, which is computed
            using the ``get_features_metadata_for_embedding`` utility function from ``teras.utils``.
            You must use the ``.get_metadata()`` method of the ``CTGANDataTransformer`` to access it.

            You can import the ``CTGANDataTransformer`` as follows,
                >>> from teras.preprocessing import CTGANDataTransformer

        units_values: ``List[int]`` or ``Tuple[int]``, default [256, 256],
            A list or tuple of units.
            For each value, a ```CTGANGeneratorBlock``` of that
            dimensionality (units) is added to the ``CTGANGeneratorBlock``
            to form its ``hidden block``.
            You can access the ``CTGANGeneratorBlock`` as follows,
                >>> from teras.layers import CTGANGeneratorBlock
    """
    def __init__(self,
                 data_dim: int,
                 metadata: dict = None,
                 units_values: UnitsValuesType = (256, 256),
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Generator residual blocks and the dimensionality of those blocks.
                        Received: {type(units_values)}""")

        if metadata is None:
            raise ValueError(f"""`metadata` cannot be None.
                `metadata`, which is computed during the data transformation step,
                is required by the `CTGANGenerator` to apply relevant activation functions to the 
                output of the Generator. But received `None`.\n
                Please pass the metadata by accessing it through the `.get_metadata()` method
                of the `CTGANDataTransformer` instance which was used to transform the raw input data.
                """)
        self.data_dim = data_dim
        self.metadata = metadata
        self.units_values = units_values

        self.hidden_block = keras.models.Sequential(name="generator_hidden_block")
        for units in self.units_values:
            self.hidden_block.add(CTGANGeneratorBlock(units))
        self.output_layer = keras.layers.Dense(self.data_dim, name="generator_output_layer")
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
        data transformation and features metadata.
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
        numerical_features_relative_indices = self.metadata["numerical"]["relative_indices_all"]
        features_relative_indices_all = self.metadata["relative_indices_all"]
        num_valid_clusters_all = self.metadata["numerical"]["num_valid_clusters_all"]
        cont_i = 0
        cat_i = 0
        num_categories_all = self.metadata["categorical"]["num_categories_all"]
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

    def get_config(self):
        config = super().get_config()
        config.update({'data_dim': self.data_dim,
                       'units_values': self.units_values,
                       'metadata': self.metadata}
                      )
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim,
                   **config)


@keras.saving.register_keras_serializable(package="teras.models")
class CTGANDiscriminator(keras.Model):
    """
    CTGANDiscriminator for CTGAN architecture as proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        units_values: ``List[int]`` or ``Tuple[int]``, default [256, 256],
            A list or tuple of units.
            For each value, a ```CTGANDiscriminatorBlock```
            of that dimensionality (units) is added to the discriminator
            to form the `hidden block` of the discriminator.

            You can import the ``CTGANDiscriminatorBlock`` as follows,
                >>> from teras.layers import CTGANDiscriminatorBlock

        packing_degree: ``int``, default 8,
            Packing degree - taken from the PacGAN paper.
            The number of samples concatenated or "packed" together.
            It must be a factor of the batch_size.
            Packing degree is borrowed from the PacGAN [Diederik P. Kingma et al.] architecture,
            which proposes passing `m` samples at once to discriminator instead of `1` to be
            jointly classified as real or fake by the discriminator in order to tackle the
            issue of mode collapse inherent in the GAN based architectures.
            The number of samples passed jointly `m`, is termed as the `packing degree`.

        gradient_penalty_lambda: ``float``, default 10,
                Controls the strength of gradient penalty.
                lambda value is directly proportional to the strength of gradient penalty.
                Gradient penalty penalizes the discriminator for large weights in an attempt
                to combat Discriminator becoming too confident and overfitting.

    Example:
        ```python
        # Instantiate Generator
        generator = CTGANGenerator(data_dim=data_dim,
                                   metadata=metadata)

        # Instantiate Discriminator
        discriminator = CTGANDiscriminator()

        # Sample noise to generate samples from
        z = tf.random.normal([512, 18])

        # Generate samples
        generated_samples = generator(z)

        # Predict using discriminator
        y_pred = discriminator(generated_samples)
        ```
    """
    def __init__(self,
                 units_values: UnitsValuesType = (256, 256),
                 packing_degree: int = 8,
                 gradient_penalty_lambda: float = 10,
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Discriminator blocks and the dimensionality of those blocks.
                        But {units_values} was passed""")

        self.units_values = units_values
        self.packing_degree = packing_degree
        self.gradient_penalty_lambda = gradient_penalty_lambda

        self.hidden_block = keras.models.Sequential(name="discriminator_hidden_block")
        for units in self.units_values:
            self.hidden_block.add(CTGANDiscriminatorBlock(units))

        self.output_layer = keras.layers.Dense(1, name="discriminator_output_layer")

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

    def get_config(self):
        config = super().get_config()
        new_config = {'units_values': self.units_values,
                      'packing_degree': self.packing_degree,
                      'gradient_penalty_lambda': self.gradient_penalty_lambda}
        config.update(new_config)
        return config


@keras.saving.register_keras_serializable(package="teras.models")
class CTGAN(_CTGAN_LF):
    """
    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        input_dim: ``int``,
            The dimensionality of the input dataset.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionailty of the transformed dataset throught the
            ``.data_dim`` attribute of the ``CTGANDataSampler`` instance used in sampling
            the dataset.

        metadata: ``dict``,
            A dictionary of metadata computed during the data transformation step.
            Simply pass the result of ``.get_metadata()`` method of the ``CTGANDataTransformer`` instance
            which was used to transform the raw input data.
            The Generator in ``CTGAN`` architecture applies different activation functions
            to the output of Generator, depending on the type of features.
            And to determine the feature types and for other computation during
            activation step, the ``metadata`` computed during the data transformation step,
            is required.
            It is also required during the computation of generator loss.
            Hence, it cannot be None.

        generator_units_values: ``List[int]`` or ``Tuple[int]``, default [256, 256],
            A list or tuple of units.
            For each value, a ``CTGANGeneratorBlock``
            of that dimensionality (units) is added to the generator
            to form the ``hidden block`` of the generator.
            You can import the ``CTGANGeneratorBlock`` as follows,
                >>> from teras.layers import CTGANGeneratorBlock

        discriminator_units_values: ``List[int]`` or ``Tuple[int]``, default [256, 256],
            A list or tuple of units values.
            For each value, a ``CTGANDiscriminatorBlock``
            of that dimensionality (units) is added to the discriminator
            to form the ``hidden block`` of the discriminator.
            You can import the ``CTGANDiscriminatorBlock`` as follows,
                >>> from teras.layers import CTGANDiscriminatorBlock

        num_discriminator_steps: ``int``, default 1,
            Number of discriminator training steps per ``CTGAN`` training step.

        latent_dim: ``int``, default 128,
            Dimensionality of noise or `z` that serves as input to Generator
            to generate samples.

        packing_degree: ``int``, default 8,
            Packing degree - taken from the PacGAN paper.
            The number of samples concatenated or "packed" together.
            It must be a factor of the batch_size.
            Packing degree is borrowed from the PacGAN [Diederik P. Kingma et al.] architecture,
            which proposes passing ``m`` samples at once to discriminator instead of ``1`` to be
            jointly classified as real or fake by the discriminator in order to tackle the
            issue of mode collapse inherent in the GAN based architectures.
            The number of samples passed jointly ``m``, is termed as the ``packing degree``.

        gradient_penalty_lambda: ``float``, default 10,
            Controls the strength of gradient penalty in the ``CTGANDiscriminator``.
            lambda value is directly proportional to the strength of gradient penalty.
            Gradient penalty penalizes the discriminator for large weights in an attempt
            to combat Discriminator becoming too confident and overfitting.
    """
    def __init__(self,
                 input_dim: int,
                 metadata: dict = None,
                 generator_units_values: UnitsValuesType = (256, 256),
                 discriminator_units_values: UnitsValuesType = (256, 256),
                 num_discriminator_steps: int = 1,
                 latent_dim: int = 128,
                 packing_degree: int = 8,
                 gradient_penalty_lambda: float = 10,
                 **kwargs):
        if metadata is None:
            raise ValueError("`metadata` cannot be None. "
                             "You can access the `metadata` through `.get_metadata()` method of `CTGANDataTransformer` "
                             "instance.")
        generator = CTGANGenerator(data_dim=input_dim,
                                   metadata=metadata,
                                   units_values=generator_units_values)

        discriminator = CTGANDiscriminator(units_values=discriminator_units_values,
                                           packing_degree=packing_degree,
                                           gradient_penalty_lambda=gradient_penalty_lambda)
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         num_discriminator_steps=num_discriminator_steps,
                         latent_dim=latent_dim,
                         **kwargs)
        self.input_dim = input_dim
        self.metadata = metadata
        self.generator_units_values = generator_units_values
        self.discriminator_units_values = discriminator_units_values
        self.num_discriminator_steps = num_discriminator_steps
        self.latent_dim = latent_dim
        self.packing_degree = packing_degree
        self.gradient_penalty_lambda = gradient_penalty_lambda

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'metadata': self.metadata,
                  'generator_units_values': self.generator_units_values,
                  'discriminator_units_values': self.discriminator_units_values,
                  'num_discriminator_steps': self.num_discriminator_steps,
                  'latent_dim': self.latent_dim,
                  'packing_degree': self.packing_degree,
                  'gradient_penalty_lambda': self.gradient_penalty_lambda,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim,
                   **config)
