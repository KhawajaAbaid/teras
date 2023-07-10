from tensorflow import keras
from tensorflow.keras import layers, models
from teras.models.gain import (Generator as _BaseGenerator,
                               Discriminator as _BaseDiscriminator)
from typing import List, Union

LAYER_OR_MODEL = Union[keras.layers.Layer, List[keras.layers.Layer], keras.models.Model]


class Generator(_BaseGenerator):
    """
    GAIN Generator model with LayerFlow design,
    for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper,
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        hidden_block: `layers.Layer`, `List[layers.Layer]` or `models.Model`,
            An instance of `GAINGeneratorBlock`, List of or Keras model made up of
            `GAINGeneratorBlock` layers and/or any custom layers
            that can work as a hidden block for the Generator model.
            If None, a hidden block with default number of `GAINGeneratorBlock` layers with
            default values will be used.
            You can import the `GAINGeneratorBlock` layer as follows,
                >>> from teras.layerflow.layers import GAINGeneratorBlock
        output_layer: `layers.Layer`,
            An instance of keras Dense layer or any custom layer that can serve as the output
            layer in the GAIN Generator model.
            It must have the dimensionality equal to `data_dim`, which is the dimensionality
            of the transformed dataset that is passed to the `GAIN` fit method and not
            necessarily equal to the dimensions of the raw input dataset.
            By default, a simple `Dense` layer of `data_dim` dimensionality
            wit "sigmoid" activation is used as the output layer.
    """
    def __init__(self,
                 hidden_block: LAYER_OR_MODEL = None,
                 output_layer: keras.layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            if isinstance(hidden_block, (layers.Layer, models.Model)):
                # leave it as is
                hidden_block = hidden_block
            elif isinstance(hidden_block, (list, tuple)):
                hidden_block = models.Sequential(hidden_block,
                                                 name="generator_hidden_block")
            else:
                raise TypeError("`hidden_block` can either be a Keras layer, list of layers or a keras model "
                                f"but received type: {type(hidden_block)} which is not supported.")
            self.hidden_block = hidden_block

        if output_layer is not None:
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
    GAIN Discrimintor model with LayerFlow design,
    for the GAIN architecture proposed by
    Jinsung Yoon et al. in the paper,
    GAIN: Missing Data Imputation using Generative Adversarial Nets.

    Reference(s):
        https://arxiv.org/abs/1806.02920

    Args:
        hidden_block: `layers.Layer`, `List[layers.Layer]` or `models.Model`,
            An instance of `GAINDiscriminatorBlock`, List of or Keras model made up of
            `GAINDiscriminatorBlock` layers and/or any custom layers
            that can work as a hidden block for the Discriminator model.
            If None, a hidden block with default number of `GAINDiscriminatorBlock` layers with
            default values will be used.
            You can import the `GAINDiscriminatorBlock` layer as follows,
                >>> from teras.layerflow.layers import GAINDiscriminatorBlock
        output_layer: `layers.Layer`,
            An instance of keras Dense layer or any custom layer that can serve as the output
            layer in the GAIN Discriminator model.
            It must have the dimensionality equal to `data_dim`, which is the dimensionality
            of the transformed dataset that is passed to the `GAIN` fit method and not
            necessarily equal to the dimensions of the raw input dataset.
            By default, a simple `Dense` layer of `data_dim` dimensionality
            wit "sigmoid" activation is used as the output layer.
    """
    def __init__(self,
                 hidden_block: LAYER_OR_MODEL = None,
                 output_layer: keras.layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            if isinstance(hidden_block, (layers.Layer, models.Model)):
                # leave it as is
                hidden_block = hidden_block
            elif isinstance(hidden_block, (list, tuple)):
                hidden_block = models.Sequential(hidden_block,
                                                 name="discriminator_hidden_block")
            else:
                raise TypeError("`hidden_block` can either be a Keras layer, list of layers or a keras model "
                                f"but received type: {type(hidden_block)} which is not supported.")
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config


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
        generator: `keras.Model`,
            An instance of `GAINGenerator` model or any customized model that can
            work in its place.
            If None, a default instance of `GAINGenerator` will be used.
            This allows you to take full control over the Generator's architecture.
            You import the standalone `GAINGenerator` model as follows,
                >>> from teras.layerflow.models import GAINGenerator
        discriminator: `keras.Model`,
            An instance of `GAINDiscriminator` model or any customized model that
            can work in its place.
            If None, a default instance of `GAINDiscriminator` will be used.
            This allows you to take full control over the Discriminator's architecture.
            You import the standalone `GAINDiscriminator` model as follows,
                >>> from teras.layerflow.models import GAINDiscriminator
        num_discriminator_steps: `int`, default 1,
            Number of discriminator training steps per GAIN training step.
        hint_rate: `float`, default 0.9,
            Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Must be between 0. and 1.
            Hint vectors ensure that generated samples follow the
            underlying data distribution.
        alpha: `float`, default 100,
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
                 generator: Generator = None,
                 discriminator: Discriminator = None,
                 data_dim: int = None,
                 num_discriminator_steps: int = 1,
                 hint_rate: float = 0.9,
                 alpha: float = 100,
                 **kwargs):
        if (generator is None and discriminator is None) and data_dim is None:
            raise ValueError(f"""`data_dim` is required to instantiate the Generator and Discriminator objects,
            if the `generator` and `discriminator` arguments are not specified.
            You can either pass the value for `data_dim` -- which can be accessed through `.data_dim`
            attribute of DataSampler instance if you don't know the data dimensions --
            or you can instantiate and pass your own `Generator` and `Discriminator` instances,
            in which case you can leave the `data_dim` parameter as None.""")
        if data_dim is None:
            data_dim = generator.data_dim if generator is not None else discriminator.data_dim
        super().__init__(data_dim=data_dim,
                         num_discriminator_steps=num_discriminator_steps,
                         hint_rate=hint_rate,
                         alpha=alpha,
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
