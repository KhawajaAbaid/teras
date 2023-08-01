from tensorflow import keras
from teras.layers.gain import GAINGeneratorBlock, GAINDiscriminatorBlock
from teras.layerflow.models.gain import GAIN as _GAIN_LF
from teras.utils.types import UnitsValuesType, ActivationType


@keras.saving.register_keras_serializable(package="teras.models")
class GAINGenerator(keras.Model):
    """
    Generator model for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        data_dim: ``int``,
            The dimensionality of the input dataset.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionality of the transformed dataset through the
            ``.data_dim`` attribute of the ``GAINDataSampler`` instance used in sampling
            the dataset.

        units_values: ``List[int]`` or ``Tuple[int]``,
            A list or tuple of units for constructing hidden block.
            For each value, a ``GAINGeneratorBlock``
            (``from teras.layers import GAINGeneratorBlock``)
            of that dimensionality (units) is added to the generator
            to form the ``hidden block`` of the generator.
            By default, ``units_values`` = [``data_dim``, ``data_dim``].

        activation_hidden: default "relu",
            Activation function to use for the hidden layers in the hidden block.

        activation_out: default "sigmoid",
            Activation function to use for the output layer of the Generator.

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
                 units_values: UnitsValuesType = None,
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = "sigmoid",
                 **kwargs):
        super().__init__(**kwargs)

        if units_values is not None and not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Generator blocks and the dimensionality of those blocks.
                        But received type {type(units_values)} which is not supported.""")

        self.data_dim = data_dim
        self.units_values = units_values
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out

        if self.units_values is None:
            self.units_values = [self.data_dim] * 2
        self.hidden_block = keras.models.Sequential(name="generator_hidden_block")
        for units in self.units_values:
            self.hidden_block.add(GAINGeneratorBlock(units,
                                                     activation=self.activation_hidden))

        self.output_layer = keras.layers.Dense(self.data_dim,
                                               activation=self.activation_out,
                                               name="generator_output_layer")

    def call(self, inputs):
        # inputs is the concatenation of `mask` and `original data`
        # where `mask` has the same dimensions as data
        # so the inputs received are 2x the dimensions of `original data`
        outputs = self.hidden_block(inputs)
        outputs = self.output_layer(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        activation_hidden_serialized = self.activation_hidden
        if not isinstance(self.activation_hidden, str):
            activation_hidden_serialized = keras.layers.serialize(self.activation_hidden)

        activation_out_serialized = self.activation_out
        if not isinstance(self.activation_out, str):
            activation_out_serialized = keras.layers.serialize(self.activation_out)

        config.update({'data_dim': self.data_dim,
                       "units_values": self.units_values,
                       "activation_hidden": activation_hidden_serialized,
                       "activation_out": activation_out_serialized,
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim,
                   **config)


@keras.saving.register_keras_serializable(package="teras.models")
class GAINDiscriminator(keras.Model):
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
        data_dim: ``int``,
            The dimensionality of the input dataset.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionality of the transformed dataset through the
            ``.data_dim`` attribute of the ``GAINDataSampler`` instance used in sampling
            the dataset.

        units_values: ``List[int]`` or ``Tuple[int]``,
            A list or tuple of units for constructing hidden block.
            For each value, a ``GAINDiscriminatorBlock``
            (``from teras.layers import GAINDiscriminatorBlock``)
            of that dimensionality (units) is added to the discriminator
            to form the ``hidden block`` of the discriminator.
            By default, ``units_values`` = [``data_dim``, ``data_dim``].

        activation_hidden: default "relu",
            Activation function to use for the hidden layers in the hidden block.

        activation_out: default "sigmoid",
            Activation function to use for the output layer of the Discriminator.

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
                 units_values: UnitsValuesType = None,
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = "sigmoid",
                 **kwargs):
        super().__init__(**kwargs)

        if units_values is not None and not isinstance(units_values, (list, tuple)):
            raise ValueError(f"""`units_values` must be a list or tuple of units which determines
                        the number of Discriminator blocks and the dimensionality of those blocks.
                        But {units_values} was passed.""")

        self.data_dim = data_dim
        self.units_values = units_values
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out

        if self.units_values is None:
            self.units_values = [self.data_dim] * 2
        self.hidden_block = keras.models.Sequential(name="discriminator_hidden_block")
        for units in self.units_values:
            self.hidden_block.add(GAINDiscriminatorBlock(units,
                                                         activation=self.activation_hidden))

        self.output_layer = keras.layers.Dense(self.data_dim,
                                               activation=self.activation_out,
                                               name="discriminator_output_layer")

    def call(self, inputs):
        # inputs is the concatenation of `hint` and manipulated
        # Generator output (i.e. generated samples).
        # `hint` has the same dimensions as data
        # so the inputs received are 2x the dimensions of original data
        outputs = self.hidden_block(inputs)
        outputs = self.output_layer(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        activation_hidden_serialized = self.activation_hidden
        if not isinstance(self.activation_hidden, str):
            activation_hidden_serialized = keras.layers.serialize(self.activation_hidden)

        activation_out_serialized = self.activation_out
        if not isinstance(self.activation_out, str):
            activation_out_serialized = keras.layers.serialize(self.activation_out)

        config.update({'data_dim': self.data_dim,
                       "units_values": self.units_values,
                       "activation_hidden": activation_hidden_serialized,
                       "activation_out": activation_out_serialized,
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim,
                   **config)


@keras.saving.register_keras_serializable(package="teras.models")
class GAIN(_GAIN_LF):
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
        input_dim: ``int``,
            The dimensionality of the input dataset.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionality of the transformed dataset through the
            ``.data_dim`` attribute of the ``GAINDataSampler`` instance used in sampling
            the dataset.

        generator_units_values: ``List[int]`` or ``Tuple[int]``,
            A list or tuple of units for constructing hidden block for the Generator.
            For each value, a ``GAINGeneratorBlock``
            (``from teras.layers import GAINGeneratorBlock``)
            of that dimensionality (units) is added to the generator to form the
            `hidden block` of the generator.
            By default, ``generator_units_values = [data_dim, data_dim]``.

        discriminator_units_values: ``List[int]`` or ``Tuple[int]``,
            A list or tuple of units for constructing hidden block for the Discriminator.
            For each value, a ``GAINDiscriminatorBlock``
            (``from teras.layers import GAINDiscriminatorBlock``)
            of that dimensionality (units) is added to the discriminator
            to form the `hidden block` of the discriminator.
            By default, ``discriminator_units_values = [data_dim, data_dim]``.

        generator_activation_hidden: default "relu",
            Activation function to use for the hidden layers in the hidden block
            for the Generator.

        discriminator_activation_hidden: default "relu",
            Activation function to use for the hidden layers in the hidden block
            for the Discriminator.

        generator_activation_out: default "sigmoid",
            Activation function to use for the output layer of the Generator.

        discriminator_activation_out: default "sigmoid",
            Activation function to use for the output layer of the Discriminator.

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
        gain_imputer = GAIN(input_dim=data_sampler.data_dim)

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
                 input_dim: int,
                 generator_units_values: UnitsValuesType = None,
                 discriminator_units_values: UnitsValuesType = None,
                 generator_activation_hidden: ActivationType = "relu",
                 discriminator_activation_hidden: ActivationType = "relu",
                 generator_activation_out: ActivationType = "sigmoid",
                 discriminator_activation_out: ActivationType = "sigmoid",
                 num_discriminator_steps: int = 1,
                 hint_rate: float = 0.9,
                 alpha: float = 100,
                 **kwargs):
        generator = GAINGenerator(data_dim=input_dim,
                                  units_values=generator_units_values,
                                  activation_hidden=generator_activation_hidden,
                                  activation_out=generator_activation_out)
        discriminator = GAINDiscriminator(data_dim=input_dim,
                                          units_values=discriminator_units_values,
                                          activation_hidden=discriminator_activation_hidden,
                                          activation_out=discriminator_activation_out)
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         num_discriminator_steps=num_discriminator_steps,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         **kwargs)
        self.input_dim = input_dim
        self.generator_units_values = generator_units_values
        self.discriminator_units_values = discriminator_units_values
        self.generator_activation_hidden = generator_activation_hidden
        self.discriminator_activation_hidden = discriminator_activation_hidden
        self.generator_activation_out = generator_activation_out
        self.discriminator_activation_out = discriminator_activation_out
        self.num_discriminator_steps = num_discriminator_steps
        self.hint_rate = hint_rate
        self.alpha = alpha

    def get_config(self):
        config = {"name": self.name,
                  "trainable": self.trainable,
                  "input_dim": self.input_dim,
                  "generator_units_values": self.generator_units_values,
                  "discriminator_units_values": self.discriminator_units_values,
                  "generator_activation_hidden": self.generator_activation_hidden,
                  "discriminator_activation_hidden": self.discriminator_activation_hidden,
                  "generator_activation_out": self.generator_activation_out,
                  "discriminator_activation_out":  self.discriminator_activation_out,
                  "num_discriminator_steps": self.num_discriminator_steps,
                  "hint_rate": self.hint_rate,
                  "alpha": self.alpha,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim,
                   **config)
