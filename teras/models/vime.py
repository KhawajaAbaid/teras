import tensorflow as tf
from tensorflow import keras
from teras.layers import (VimeEncoder,
                          VimeMaskEstimator,
                          VimeFeatureEstimator,
                          VimePredictor,
                          VimeMaskGenerationAndCorruption)
from teras.utils.types import ActivationType
from teras.layerflow.models.vime import (VimeSelf as _VimeSelfLF,
                                         VimeSemi as _VimeSemiLF)


@keras.saving.register_keras_serializable(package="teras.models")
class VimeSelf(_VimeSelfLF):
    """Self-supervised learning part in the paper
    "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"
    by Jinsung Yoon et al.

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        encoder_activation: default "relu",
            Activation to use for encoder

        feature_estimator_activation: default "sigmoid",
            Activation to use for feature estimator

        mask_estimator_activation: default "sigmoid",
            Activation to use for mask estimator
    """
    def __init__(self,
                 input_dim: int,
                 encoder_activation: ActivationType = "relu",
                 feature_estimator_activation: ActivationType = "sigmoid",
                 mask_estimator_activation: ActivationType = "sigmoid",
                 **kwargs
                 ):
        encoder = VimeEncoder(input_dim,
                              input_shape=(input_dim,),
                              activation=encoder_activation,
                              name="encoder")
        # Mask estimator
        mask_out = VimeMaskEstimator(input_dim,
                                     activation=mask_estimator_activation,
                                     name="mask_estimator")
        # Feature estimator
        feature_out = VimeFeatureEstimator(input_dim,
                                           activation=feature_estimator_activation,
                                           name="feature_estimator")
        super().__init__(**kwargs)
        self.encoder_activation = encoder_activation
        self.feature_estimator_activation = feature_estimator_activation
        self.mask_estimator_activation = mask_estimator_activation

    def build(self, input_shape):
        """Builds layers and functional model"""
        _, input_dim = input_shape
        inputs = keras.layers.Input(shape=(input_dim,))
        # Encoder



    def get_config(self):
        config = super().get_config()

        encoder_activation_serialized = self.encoder_activation
        if not isinstance(encoder_activation_serialized, str):
            encoder_activation_serialized = keras.layers.serialize(encoder_activation_serialized)

        feature_estimator_activation_serialized = self.feature_estimator_activation
        if not isinstance(feature_estimator_activation_serialized, str):
            feature_estimator_activation_serialized = keras.layers.serialize(feature_estimator_activation_serialized)

        mask_estimator_activation_serialized = self.mask_estimator_activation
        if not isinstance(mask_estimator_activation_serialized, str):
            mask_estimator_activation_serialized = keras.layers.serialize(mask_estimator_activation_serialized)

        config.update({'input_dim': self.input_dim,
                       'p_m': self.p_m,
                       'encoder_activation': self.encoder_activation,
                       'feature_estimator_activation': self.feature_estimator_activation,
                       'mask_estimator_activation': self.mask_estimator_activation,
                       }
                      )
        return config


@keras.saving.register_keras_serializable(package="teras.models")
class VimeSemi(_VimeSemiLF):
    """
    Semi-supervised learning part of the VIME architecture,
    proposed by Jinsung Yoon et al. in the paper
    "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset.

        pretrained_encoder: ``keras.Model``,
            Pretrained instance of ``VimEncoder``, pretrained during the
            Self-Supervised learning phase.
            You can access it after training the ``VimeSelf`` model through its
            ``.get_encoder()`` method.

        num_labels: ``int``, default 1,
            Number of labels to predict.

        hidden_dim: ``int``, default 32,
            Dimensionality of hidden layers in the ``VimePredictor`` layer.

        p_m: ``float``, default 0.3,
            Corruption probability

        K: ``int``, default 3,
            Number of augmented samples

        beta: ``float``, default 1.0,
            Hyperparameter to control supervised and unsupervised loss
    """
    def __init__(self,
                 input_dim: int,
                 pretrained_encoder: keras.Model,
                 num_labels: int = 2,
                 hidden_dim: int = 32,
                 p_m: float = 0.3,
                 K: int = 3,
                 beta: float = 1.0,
                 **kwargs):
        mask_and_corrupt = VimeMaskGenerationAndCorruption(p_m,
                                                           name="mask_generation_and_corruption")
        predictor = VimePredictor(num_labels=num_labels,
                                  hidden_dim=hidden_dim,
                                  name="predictor",
                                  )
        super().__init__(input_dim=input_dim,
                         pretrained_encoder=pretrained_encoder,
                         mask_generation_and_corruption=mask_and_corrupt,
                         predictor=predictor,
                         K=K,
                         beta=beta,
                         **kwargs)
        self.input_dim = input_dim
        self.encoder = pretrained_encoder
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.p_m = p_m
        self.K = K
        self.beta = beta

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'pretrained_encoder': keras.layers.serialize(self.pretrained_encoder),
                       'num_labels': self.num_labels,
                       'hidden_dim': self.hidden_dim,
                       'p_m': self.p_m,
                       'K': self.K,
                       'beta': self.beta,
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        pretrained_encoder = keras.layers.deserialize(config.pop("pretrained_encoder"))
        return cls(input_dim=input_dim,
                   pretrained_encoder=pretrained_encoder,
                   **config)
