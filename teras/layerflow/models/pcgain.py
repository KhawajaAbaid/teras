from tensorflow.keras import layers, models
from tensorflow import keras
from teras.models.pcgain import (Classifier as _BaseClassifier,
                                 PCGAIN as _BasePCGAIN)
from teras.layerflow.models import GAIN
from typing import List, Tuple, Union


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
INT_OR_FLOAT = Union[int, float]
HIDDEN_BLOCK_TYPE = Union[keras.layers.Layer, keras.models.Model]


class Classifier(_BaseClassifier):
    """
    The auxiliary classifier for the PC-GAIN architecture
    proposed by Yufeng Wang et al. in the paper
    "PC-GAIN: Pseudo-label Conditional Generative Adversarial
    Imputation Networks for Incomplete Data"

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        hidden_block: `layers.Layer`, `List[layers.Layer]` or `models.Model`,
            A Keras layer, List of layers or Keras model that can work as the
            hidden block for the Classifier model.
            If None, a hidden block made up of dense layers of `data_dim`
            dimensionality is used.
            If you specify a hidden block, `data_dim` parameter is ignored.
        output_layer: `layers.Layer`,
            An instance of keras Dense layer or any custom layer that can serve as the output
            layer in the GAIN Discriminator model.
            If you specify an output layer, `num_classes` parameter is ignored.
        num_classes: Number of classes to predict.
            It should be equal to the `num_clusters`,
            computed during the pseudo label generation.
    """
    def __init__(self,
                 hidden_block: HIDDEN_BLOCK_TYPE = None,
                 output_layer: keras.layers.Layer = None,
                 data_dim: int = None,
                 num_classes: int = None,
                 **kwargs):
        if hidden_block is None and data_dim is None:
            raise ValueError("`hidden_block` and `data_dim` both cannot be None at the same time. "
                             "You must either specify a hidden block or pass value for `data_dim` so that a default "
                             "hidden block can be constructed. ")
        if output_layer is None and num_classes is None:
            raise ValueError("`output_layer` and `num_classes` both cannot be None at the same time. "
                             "You must either specify an output layer or pass value for `num_classes` so that a "
                             "default output layer can be constructed.")
        # If user passes a hidden block then in that case we ignore the data_dim,
        # but since data_dim is required by the parent layer so we pass a random value
        # -- since it'll be overriden by hidden block later.
        # Same goes for the num_classes.
        if data_dim is None:
            data_dim = 16
        if num_classes is None:
            num_classes = 3
        super().__init__(data_dim=data_dim,
                         num_classes=num_classes,
                         **kwargs)

        self.hidden_block = hidden_block
        self.output_layer = output_layer
        self.data_dim = data_dim
        self.num_classes = num_classes

        if hidden_block is not None:
            if not isinstance(hidden_block, (layers.Layer, models.Model)):
                raise TypeError("`hidden_block` can either be a Keras layer, or a Keras model "
                                f"but received type: {type(hidden_block)} which is not supported.")
            self.hidden_block = hidden_block

    def call(self, inputs):
        x = self.hidden_block(inputs)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config


class PCGAIN(_BasePCGAIN):
    """
    PCGAIN model with LayerFlow design.
    PCGAIN is a missing data imputation model based
    on the GAIN architecture.

    This implementation is based on the architecture
    proposed by Yufeng Wang et al. in the paper
    "PC-GAIN: Pseudo-label Conditional Generative Adversarial
    Imputation Networks for Incomplete Data"

    Reference(s):
        https://arxiv.org/abs/2011.07770

    Args:
        generator: `keras.Model`,
            An instance of `PCGAINGenerator` model or any customized model that can
            work in its place.
            If None, a default instance of `PCGAINGenerator` will be used.
            This allows you to take full control over the Generator's architecture.
            You import the standalone `PCGAINGenerator` model as follows,
                >>> from teras.layerflow.models import PCGAINGenerator

        discriminator: `keras.Model`,
            An instance of `PCGAINDiscriminator` model or any customized model that
            can work in its place.
            If None, a default instance of `PCGAINDiscriminator` will be used.
            This allows you to take full control over the Discriminator's architecture.
            You import the standalone `PCGAINDiscriminator` model as follows,
                >>> from teras.layerflow.models import PCGAINDiscriminator

        pretrainer: `keras.Model`,
            An instance of `GAIN` model or any custom keras model that can be used
            as a pretrainer to pretrain the `Generator` and `Discrimintor` instances.
            The pretrainer is used to pretrain the Generator and Discriminators, and to impute
            the pretraining dataset, which is then clustered to generated pseudo labels
            which combined with the imputed data are used to train the classifier.

            IMPORTANT NOTE:
                If you pass a pretrainer instance, the `generator` and `discriminator` arguments
                will be ignored as the Generator and Discriminator instances being used by
                the `pretrainer` will be used by the PCGAIN model.

        classifier: `keras.Model`,
            An instance of the `PCGAINClassifier` model or any custom model that can work
            in its palce.
            It is used in the training of the Generator component
            of the PCGAIN architecture after it has been pretrained by the `pretrainer`.
            The classifier itself is trained on the imputed data in the pretraining step
            combined with the pseudo labels generated for the imputed data by clustering.
            You can import the `PCGAINClassifier` as follows,
                >>> from teras.layerflow.models import PCGAINClassifier
        num_discriminator_steps: default 1, Number of discriminator training steps
            per PCGAIN training step.
        hint_rate: Hint rate will be used to sample binary vectors for
            `hint vectors` generation. Should be between 0. and 1.
            Hint vectors ensure that generated samples follow the
            underlying data distribution.
        alpha: Hyper parameter for the generator loss computation that
            controls how much weight should be given to the MSE loss.
            Precisely, `generator_loss` = `cross_entropy_loss` + `alpha` * `mse_loss`
            The higher the `alpha`, the more the mse_loss will affect the
            overall generator loss.
            Defaults to 200.
        beta: Hyper parameter for generator loss computation that
            controls the contribution of the classifier's loss to the
            overall generator loss.
            Defaults to 100.
        num_clusters: Number of clusters to cluster the imputed data
            that is generated during pretraining.
            These clusters serve as pseudo labels for training of classifier.
            Defaults to 5.
        clustering_method: Should be one of the following,
            ["Agglomerative", "KMeans", "MiniBatchKMeans", "Spectral", "SpectralBiclustering"]
            The names are case in-sensitive.
            Defaults to "kmeans"
    """
    def __init__(self,
                 generator: keras.Model = None,
                 discriminator: keras.Model = None,
                 pretrainer: keras.Model = None,
                 classifier: keras.Model = None,
                 data_dim: int = None,
                 num_discriminator_steps: int = 1,
                 hint_rate: float = 0.9,
                 alpha: INT_OR_FLOAT = 200,
                 beta: INT_OR_FLOAT = 100,
                 num_clusters: int = 5,
                 clustering_method: str = "kmeans",
                 **kwargs):
        super().__init__(**kwargs)

        if pretrainer is None:
            if (generator is None and discriminator is None) and data_dim is None:
                raise ValueError(f"""`data_dim` is required to instantiate the Generator and Discriminator objects,
                if the `generator` and `discriminator` arguments are not specified.
                You can either pass the value for `data_dim` -- which can be accessed through `.data_dim`
                attribute of DataSampler instance if you don't know the data dimensions --
                or you can instantiate and pass your own `Generator` and `Discriminator` instances,
                in which case you can leave the `data_dim` parameter as None.""")

            if data_dim is None:
                data_dim = generator.data_dim if generator is not None else discriminator.data_dim

        if data_dim is None:
            # fill it with a random value
            data_dim = 16
        super().__init__(data_dim=data_dim,
                         num_discriminator_steps=num_discriminator_steps,
                         hint_rate=hint_rate,
                         alpha=alpha,
                         beta=beta,
                         num_clusters=num_clusters,
                         clustering_method=clustering_method,
                         **kwargs)

        if pretrainer is not None:
            # If user has passed a pretrainer it must have generator and discriminator instances as attributes
            # which it will pretrian.
            # We use the same generator and discriminator instances from the pretrainer.
            self.pretrainer = pretrainer
            self.generator = pretrainer.generator
            self.discriminator = pretrainer.discriminator
        else:
            # If user doesn't pass a pretrainer, we set values for generator and discriminator if they exist
            # and then create a pretrainer instance -- using GAIN, not PCGAIN, as our pretrainer.
            if generator is not None:
                self.generator = generator
            if discriminator is not None:
                self.discriminator = discriminator
            self.pretrainer = GAIN(generator=self.generator,
                                   discriminator=self.discriminator,
                                   num_discriminator_steps=self.num_discriminator_steps,
                                   hint_rate=self.hint_rate,
                                   alpha=self.alpha)

        if classifier is not None:
            self.classifier = classifier

    def get_config(self):
        config = super().get_config()
        new_config = {'generator': keras.layers.serialize(self.generator),
                      'discriminator': keras.layers.serialize(self.discriminator),
                      'pretrainer': keras.layers.serialize(self.pretrainer),
                      'classifier': keras.layers.serialize(self.classifier),
                      }
        config.update(new_config)
        return config
