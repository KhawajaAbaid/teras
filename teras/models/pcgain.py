from tensorflow import keras
from teras.models.gain import (GAINGenerator as _GAINGenerator,
                               GAINDiscriminator as _GAINDiscriminator,
                               GAIN)
from teras.utils.types import UnitsValuesType, ActivationType
from teras.layerflow.models.pcgain import PCGAIN as _PCGAIN_LF


@keras.saving.register_keras_serializable(package="teras.models")
class PCGAINGenerator(_GAINGenerator):
    """
    Generator model for the PCGAIN architecture.
    It is exact as the ``GAINGenerator`` model, in fact, this class is just a wrapper
    around the ``GAINGenerator`` model class.

    Args:
        data_dim: ``int``,
            The dimensionality of the input dataset.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionality of the transformed dataset through the
            ``.data_dim`` attribute of the ``PCGAINDataSampler`` instance used in sampling
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
        ```
    """
    def __init__(self,
                 data_dim: int,
                 units_values: UnitsValuesType = None,
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = "sigmoid",
                 **kwargs):
        super().__init__(data_dim=data_dim,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out,
                         **kwargs)


@keras.saving.register_keras_serializable(package="teras.models")
class PCGAINDiscriminator(_GAINDiscriminator):
    """
    Discriminator model for the PCGAIN architecture.
    It is exact as the ``GAINDiscriminator`` model, in fact, this class is just a wrapper
    around the ``GAINDiscriminator`` model class.

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
        ```
    """
    def __init__(self,
                 data_dim: int,
                 units_values: UnitsValuesType = None,
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = "sigmoid",
                 **kwargs):
        super().__init__(data_dim=data_dim,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out,
                         **kwargs)


@keras.saving.register_keras_serializable(package="teras.models")
class PCGAINClassifier(keras.Model):
    """
    The auxiliary classifier for the PC-GAIN architecture
    proposed by Yufeng Wang et al. in the paper
    "PC-GAIN: Pseudo-label Conditional Generative Adversarial
    Imputation Networks for Incomplete Data"

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        data_dim: ``int``,
            The dimensionality of the input dataset.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

        num_classes: ``int``,
            Number of classes to predict.
            It should be equal to the `num_clusters`,
            computed during the pseudo label generation.

        units_values: ``List[int]`` or ``Tuple[int]``,
            A list/tuple of units to construct hidden block of classifier.
            For each element, a new hidden layer will be added.
            By default, `units_values` = [`data_dim`, `data_dim`]

        activation_hidden: default "relu",
            Activation function to use for the hidden layers in the classifier.

        activation_out: default "softmax",
            Activation function to use for the output layer of classifier.
    """
    def __init__(self,
                 data_dim: int,
                 num_classes: int = None,
                 units_values: UnitsValuesType = None,
                 activation_hidden: ActivationType = "relu",
                 activation_out: ActivationType = "softmax",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.units_values = units_values
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out

        if self.units_values is None:
            self.units_values = [data_dim] * 2

        self.hidden_block = keras.models.Sequential(name="classifier_hidden_block")
        for units in self.units_values:
            self.hidden_block.add(keras.layers.Dense(units,
                                                     activation=self.activation_hidden))

        self.output_layer = keras.layers.Dense(self.num_classes,
                                               activation=self.activation_out)

    def call(self, inputs):
        x = self.hidden_block(inputs)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        new_config = {'data_dim': self.data_dim,
                      'num_classes': self.num_classes,
                      'units_values': self.units_values,
                      'activation_hidden': self.activation_hidden,
                      'activation_out': self.activation_out
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim,
                   **config)


@keras.saving.register_keras_serializable(package="teras.models")
class PCGAIN(_PCGAIN_LF):
    """
    PC-GAIN is a missing data imputation model based
    on the GAIN architecture.

    This implementation is based on the architecture
    proposed by Yufeng Wang et al. in the paper
    "PC-GAIN: Pseudo-label Conditional Generative Adversarial
    Imputation Networks for Incomplete Data"

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        input_dim: ``int``,
            The dimensionality of the input dataset.

            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.

            You can access the dimensionality of the transformed dataset through the
            ``.data_dim`` attribute of the ``PCGAINDataSampler`` instance used in sampling
            the dataset.

        generator_units_values: ``List[int]`` or ``Tuple[int]``,
            A list or tuple of units for constructing hidden block for the Generator.
            For each value, a ``PCGAINDiscriminatorBlock``
            (``from teras.layers import PCGAINGeneratorBlock``)
            of that dimensionality (units) is added to the generator to form the
            `hidden block` of the generator.
            By default, ``generator_units_values = [data_dim, data_dim]``.

        discriminator_units_values: ``List[int]`` or ``Tuple[int]``,
            A list or tuple of units for constructing hidden block for the Discriminator.
            For each value, a ``PCGAINDiscriminatorBlock``
            (``from teras.layers import PCGAINDiscriminatorBlock``)
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
            Number of discriminator training steps per PCGAIN training step.

        hint_rate: ``float``, default 0.9,
            Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Should be between 0. and 1.
            Hint vectors ensure that generated samples follow the
            underlying data distribution.

        alpha: ``float``, default 200.,
            Hyper parameter for the generator loss computation that
            controls how much weight should be given to the MSE loss.
            Precisely, ``generator_loss = cross_entropy_loss + alpha * mse_loss``
            The higher the ``alpha``, the more the ``mse_loss`` will affect the
            overall generator loss.

        beta: ``float``, default 100.,
            Hyper parameter for generator loss computation that
            controls the contribution of the classifier's loss to the
            overall generator loss.

        num_clusters: ``int``, default 5,
            Number of clusters to cluster the imputed data
            that is generated during pretraining.
            These clusters serve as pseudo labels for training of classifier.

        clustering_method: ``str``, default "kmeans",
            Should be one of the following,
            ["Agglomerative", "KMeans", "MiniBatchKMeans", "Spectral", "SpectralBiclustering"]
            The names are case in-sensitive.
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
                 alpha: float = 200.,
                 beta: float = 100.,
                 num_clusters: int = 5,
                 clustering_method: str = "kmeans",
                 **kwargs):
        # Since we pretrain using the EXACT SAME architecture as GAIN
        # so here we simply use the GAIN model, which acts as `pretrainer`.
        # And since it instantiates generators and discriminator models,
        # we can pass these generator and discriminator parameters to GAIN
        # which acts as a proxy pretrainer and have it instantiate and pretrain them,
        # and then we can access those pretrained Generator and Discriminator models
        # and use them here in our PC-GAIN architecture.
        pretrainer = GAIN(input_dim=input_dim,
                          generator_units_values=generator_units_values,
                          generator_activation_hidden=generator_activation_hidden,
                          generator_activation_out=generator_activation_out,
                          discriminator_units_values=discriminator_units_values,
                          discriminator_activation_hidden=discriminator_activation_hidden,
                          discriminator_activation_out=discriminator_activation_out,
                          hint_rate=hint_rate,
                          alpha=alpha
                          )
        classifier = PCGAINClassifier(data_dim=input_dim,
                                      num_classes=num_clusters)

        super().__init__(pretrainer=pretrainer,
                         classifier=classifier,
                         num_discriminator_steps=num_discriminator_steps,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         beta=beta,
                         num_clusters=num_clusters,
                         clustering_method=clustering_method,
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
        self.beta = beta
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method

    def get_config(self):
        gen_act_hidden_serialized = self.generator_activation_hidden
        if not isinstance(gen_act_hidden_serialized, str):
            gen_act_hidden_serialized = keras.layers.serialize(gen_act_hidden_serialized)

        gen_act_out_serialized = self.generator_activation_out
        if not isinstance(gen_act_out_serialized, str):
            gen_act_out_serialized = keras.layers.serialize(gen_act_out_serialized)

        disc_act_hidden_serialized = self.discriminator_activation_hidden
        if not isinstance(disc_act_hidden_serialized, str):
            disc_act_hidden_serialized = keras.layers.serialize(disc_act_hidden_serialized)

        disc_act_out_serialized = self.discriminator_activation_out
        if not isinstance(disc_act_out_serialized, str):
            disc_act_out_serialized = keras.layers.serialize(disc_act_out_serialized)

        config = {"name": self.name,
                  "trainable": self.trainable,
                  "input_dim": self.input_dim,
                  "generator_units_values": self.generator_units_values,
                  "discriminator_units_values": self.discriminator_units_values,
                  "generator_activation_hidden": gen_act_hidden_serialized,
                  "discriminator_activation_hidden": disc_act_hidden_serialized,
                  "generator_activation_out": gen_act_out_serialized,
                  "discriminator_activation_out": disc_act_out_serialized,
                  "num_discriminator_steps": self.num_discriminator_steps,
                  "hint_rate": self.hint_rate,
                  "alpha": self.alpha,
                  "beta": self.beta,
                  "num_clusters": self.num_clusters,
                  "clustering_method": self.clustering_method,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim,
                   **config)
