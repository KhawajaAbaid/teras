from tensorflow import keras
from tensorflow.keras import layers, models
from teras.models.ctgan import (Generator as _BaseGenerator,
                                Discriminator as _BaseDiscriminator,
                                CTGAN as _BaseCTGAN)
from typing import List, Union, Tuple

LIST_OR_TUPLE = Union[List[int], Tuple[int]]
HIDDEN_BLOCK_TYPE = Union[keras.layers.Layer, keras.models.Model]


class Generator(_BaseGenerator):
    """
    Generator for CTGAN architecture with LayerFlow design.
    CTGAN architecture is proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        hidden_block: `layers.Layer`, `List[layers.Layer]` or `models.Model`,
            An instance of `CTGANGeneratorBlock` layer, list of layers or Keras model
            that can work as the hidden block for the Classifier model.
            If None, a hidden block made up of `CTGANGeneratorBlock` layers with default
            dimensionality.
            You can import the `CTGANGeneratorBlock` as follows,
                >>> from teras.layerflow.layers import CTGANGeneratorBlock
        output_layer: `layers.Layer`.
            An instance of keras Dense layer or any custom layer that can serve as the output
            layer in the CTGAN Generator model.
            The output layer must be of `data_dim` dimensionality.
            By default, a simple `Dense` layer of `data_dim` dimensionality
            with no activation is used as the output layer.
        data_dim: `int`, required when `output_layer` is None.
            The dimensionality of the dataset.
            It will also be the dimensionality of the output produced
            by the generator.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
        meta_data: `dict`,
            A dictionary of features metadata.
            The Generator in CTGAN architecture,
            applies different activation functions to the output of Generator,
            depending on the type of features.
            And to determine the feature types and for other computation during
            activation step, the `meta data` computed during the data transformation step,
            is required.
            It can be accessed through the `.get_meta_data()` method of the DataTransformer
            instance which was used to transform the raw input data.
    """
    def __init__(self,
                 hidden_block: HIDDEN_BLOCK_TYPE = None,
                 output_layer: keras.layers.Layer = None,
                 data_dim: int = None,
                 meta_data: dict = None,
                 **kwargs):
        if output_layer is None and data_dim is None:
            raise ValueError("`output_layer` and `data_dim` both cannot be None at the same time as the value of "
                             "`data_dim` is required to construct a default output layer as the Generator's output "
                             "must of `data_dim` dimensionality. \n"
                             "You must either pass value for `output_layer` or `data_dim`.")
        if output_layer is not None:
            # Assign a random value to data_dim.
            # data_dim is only used in output_layer construction
            # which in this case will be overridden later
            data_dim = 16
        super().__init__(data_dim=data_dim,
                         meta_data=meta_data,
                         **kwargs)

        if hidden_block is not None:
            if not isinstance(hidden_block, (layers.Layer, models.Model)):
                raise TypeError("`hidden_block` can either be a Keras layer or a Keras model "
                                f"but received type: {type(hidden_block)} which is not supported.")
            self.hidden_block = hidden_block

        if self.output_layer is not None:
            self.output_layer = output_layer

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config


class Discriminator(_BaseDiscriminator):
    """
    Discriminator for CTGAN architecture with LayerFlow design.
    CTGAN architecture is proposed by
    Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503
            If `None`, a hidden block is constructed with `CTGANDiscriminatorBlock`
            (`from teras.layers.ctgan import DiscriminatorBlock`)
            with default dimensionality.
    Args:
        hidden_block: `layers.Layer`, `List[layers.Layer]` or `models.Model`,
            An instance of `CTGANDiscriminatorBlock` layer, list of layers or Keras model
            that can work as the hidden block for the Classifier model.
            If None, a hidden block made up of `CTGANDiscriminatorBlock` layers with default
            dimensionality will be used.
            You can import the `CTGANDiscriminatorBlock` as follows,
                >>> from teras.layerflow.layers import CTGANDiscriminatorBlock
        output_layer: `layers.Layer`,
            An instance of keras Dense layer or any custom layer that can serve as the output
            layer in the CTGAN Discriminator model.
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
                 hidden_block: HIDDEN_BLOCK_TYPE = None,
                 output_layer: keras.layers.Layer = None,
                 packing_degree: int = 8,
                 gradient_penalty_lambda=10,
                 **kwargs):
        super().__init__(packing_degree=packing_degree,
                         gradient_penalty_lambda=gradient_penalty_lambda,
                         **kwargs)

        if hidden_block is not None:
            if not isinstance(hidden_block, (layers.Layer, models.Model)):
                raise TypeError("`hidden_block` can either be a Keras layer or a Keras model "
                                f"but received type: {type(hidden_block)} which is not supported.")
            self.hidden_block = hidden_block

        if self.output_layer is not None:
            self.output_layer = output_layer

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config


class CTGAN(_BaseCTGAN):
    """
    CTGAN model with LayerFlow design.
    CTGAN is a state-of-the-art tabular data generation architecture
    proposed by Lei Xu et al. in the paper,
    "Modeling Tabular data using Conditional GAN".

    Reference(s):
        https://arxiv.org/abs/1907.00503

    Args:
        generator: `keras.Model`,
            An instance of `CTGANGenerator` or any other keras model that can work in
            its place.
            You can import the `CTGANGenerator` as follows,
                >>> from teras.layerflow.models import CTGANGenerator
        discriminator: `keras.Model`,
            An instance of `CTGANDiscriminator` or any other keras model that can work in
            its place.
            You can import the `CTGANDiscriminator` as follows,
                >>> from teras.layerflow.models import CTGANDiscriminator
        num_discriminator_steps: `int`, default 1,
            Number of discriminator training steps per CTGAN training step.
        latent_dim: `int`, default 128,
            Dimensionality of noise or `z` that serves as input to Generator
            to generate samples.
        data_dim: `int`, required only when `generator` is None,
            The dimensionality of the input dataset.
            Note the dimensionality must be equal to the dimensionality of dataset
            that is passed to the fit method and not necessarily the dimensionality
            of the raw input dataset as sometimes data transformation alters the
            dimensionality of the dataset.
        meta_data: `dict`, required only when `generator` is None,
            A dictionary of metadata about the features that is computed during the transformation phase.
            Simply pass the result of `.get_meta_data()` method of the DataTransformer instance
            which was used to transform the raw input data.
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
                 latent_dim: int = 128,
                 data_dim: int = None,
                 meta_data: dict = None,
                 **kwargs):
        if generator is None:
            if data_dim is None:
                raise ValueError(f"`data_dim` is required when `generator` argument is None, "
                                 f"to instantiate a default Generator instance. \n"
                                 "You can either specify a `generator` instance or pass the value for `data_dim`, "
                                 "which can be accessed through `.data_dim` "
                                 "attribute of DataSampler instance if you don't know the data dimensions.")
            if meta_data is None:
                raise ValueError("`meta_data` is required when `generator` argument i None, to instantiate a default "
                                 "Generator instance. "
                                 "You can either speicfy a `generator` instance or pass the meta data dictionary which "
                                 "can be accessed through `.get_meta_data()` method of DataTransformer instance.")

        if generator is not None:
            # plug random values
            # now we could also just access .data_dim and .meta_data attributes of the generator instance
            # that is passed as argument BUT that requires us to assume that only CTGANGenerator instances
            # are passed and in case user does create his/her own complete custom generator model, there's
            # no guarantee that that model will have these attributes.
            data_dim = 16
            meta_data = {'Wizard': 1337}

        super().__init__(data_dim=data_dim,
                         meta_data=meta_data,
                         num_discriminator_steps=num_discriminator_steps,
                         latent_dim=latent_dim,
                         **kwargs)

        if generator is not None:
            self.generator = generator

        if discriminator is not None:
            self.discriminator = discriminator

    def get_config(self):
        config = super().get_config()
        new_config = {'generator': keras.layers.serialize(self.generator),
                      'discriminator': keras.layers.serialize(self.discriminator)
                      }
        config.update(new_config)
        return config
