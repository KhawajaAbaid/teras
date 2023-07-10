from tensorflow import keras
from tensorflow.keras import layers, models
from teras.models import (SAINT as _BaseSAINT,
                          SAINTClassifier as _BaseSAINTClassifier,
                          SAINTRegressor as _BaseSAINTRegressor,
                          SAINTPretrainer as _BaseSAINTPretrainer)
from teras.layerflow.layers.saint import ClassificationHead, RegressionHead


class SAINT(_BaseSAINT):
    """
    SAINT architecture with LayerFlow design.
    It proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        features_metadata: `dict`,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                                                                        numerical_features,
                                                                        categorical_features)
        categorical_feature_embedding: `layers.Layer`,
            An instance of `CategoricalFeatureEmbedding` layer to embedd categorical features
            or any layer that can work in place of `CategoricalFeatureEmbedding` for that purpose.
            If None, a `CategoricalFeatureEmbedding` layer with default values will be used.
            You can import the `CategoricalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import CategoricalFeatureEmbedding

        numerical_feature_embedding: `layers.Layer`,
            An instance of `SAINTNumericalFeatureEmbedding` layer to embedd numerical features
            or any layer that can work in place of `SAINTNumericalFeatureEmbedding` for that purpose.
            If None, a `SAINTNumericalFeatureEmbedding` layer with default values will be used.
            You can import the `SAINTNumericalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import SAINTNumericalFeatureEmbedding
        encoder: `layers.Layer`,
            An instance of `SAINTEncoder` layer to encode the features embeddings,
            or any layer that can work in place of `SAINTEncoder` for that purpose.
            If None, a `SAINTEncoder` layer with default values will be used.
            You can import the `SAINTEncoder` layer as follows,
                >>> from teras.layerflow.layers import SAINTEncoder

        head: `layers.Layer`,
            An instance of ClassificationHead or RegressionHead layer for final outputs,
            or any layer that can work in place of a Head layer for that purpose.
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 saint_numerical_feature_embedding: layers.Layer = None,
                 saint_encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         **kwargs)
        if categorical_feature_embedding is not None:
            self.categorical_feature_embedding = categorical_feature_embedding

        if saint_numerical_feature_embedding is not None:
            self.numerical_feature_embedding = saint_numerical_feature_embedding

        if saint_encoder is not None:
            self.saint_encoder = saint_encoder

        if head is not None:
            self.head = head

    def get_config(self):
        config = super().get_config()
        new_config = {'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                      'saint_numerical_feature_embedding': keras.layers.serialize(self.saint_numerical_feature_embedding),
                      'saint_encoder': keras.layers.serialize(self.saint_encoder),
                      'head': keras.layers.serialize(self.head),
                      }
        config.update(new_config)
        return config


class SAINTClassifier(_BaseSAINTClassifier):
    """
    SAINTClassifier with LayerFlow design.
    It is based on the SAINT architecture proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        features_metadata: `dict`,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                                                                        numerical_features,
                                                                        categorical_features)
        categorical_feature_embedding: `layers.Layer`,
            An instance of `CategoricalFeatureEmbedding` layer to embedd categorical features
            or any layer that can work in place of `CategoricalFeatureEmbedding` for that purpose.
            If None, a `CategoricalFeatureEmbedding` layer with default values will be used.
            You can import the `CategoricalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import CategoricalFeatureEmbedding

        numerical_feature_embedding: `layers.Layer`,
            An instance of `SAINTNumericalFeatureEmbedding` layer to embedd numerical features
            or any layer that can work in place of `SAINTNumericalFeatureEmbedding` for that purpose.
            If None, a `SAINTNumericalFeatureEmbedding` layer with default values will be used.
            You can import the `SAINTNumericalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import SAINTNumericalFeatureEmbedding

        encoder: `layers.Layer`,
            An instance of `SAINTEncoder` layer to encode the features embeddings,
            or any layer that can work in place of `SAINTEncoder` for that purpose.
            If None, a `SAINTEncoder` layer with default values will be used.
            You can import the `SAINTEncoder` layer as follows,
                >>> from teras.layerflow.layers import SAINTEncoder

        head: `layers.Layer`,
            An instance of `SAINTClassificationHead` layer for the final outputs,
            or any layer that can work in place of a `SAINTClassificationHead` layer for that purpose.
            If None, `SAINTClassificationHead` layer with default values will be used.
            You can import the `SAINTClassificationHead` layer as follows,
                >>> from teras.layerflow.layers import SAINTClassificationHead
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 numerical_feature_embedding: layers.Layer = None,
                 saint_encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = ClassificationHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         numerical_feature_embedding=numerical_feature_embedding,
                         saint_encoder=saint_encoder,
                         head=head,
                         **kwargs)


class SAINTRegressor(_BaseSAINTRegressor):
    """
    SAINTClassifier with LayerFlow design.
    It is based on the SAINT architecture proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        features_metadata: `dict`,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                                                                        numerical_features,
                                                                        categorical_features)
        categorical_feature_embedding: `layers.Layer`,
            An instance of `CategoricalFeatureEmbedding` layer to embedd categorical features
            or any layer that can work in place of `CategoricalFeatureEmbedding` for that purpose.
            If None, a `CategoricalFeatureEmbedding` layer with default values will be used.
            You can import the `CategoricalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import CategoricalFeatureEmbedding

        numerical_feature_embedding: `layers.Layer`,
            An instance of `SAINTNumericalFeatureEmbedding` layer to embedd numerical features
            or any layer that can work in place of `SAINTNumericalFeatureEmbedding` for that purpose.
            If None, a `SAINTNumericalFeatureEmbedding` layer with default values will be used.
            You can import the `SAINTNumericalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import SAINTNumericalFeatureEmbedding

        encoder: `layers.Layer`,
            An instance of `SAINTEncoder` layer to encode the features embeddings,
            or any layer that can work in place of `SAINTEncoder` for that purpose.
            If None, a `SAINTEncoder` layer with default values will be used.
            You can import the `SAINTEncoder` layer as follows,
                >>> from teras.layerflow.layers import SAINTEncoder

        head: `layers.Layer`,
            An instance of `SAINTRegressionHead` layer for the final outputs,
            or any layer that can work in place of a `SAINTRegressionHead` layer for that purpose.
            If None, `SAINTRegressionHead` layer with default values will be used.
            You can import the `SAINTRegressionHead` layer as follows,
                >>> from teras.layerflow.layers import SAINTRegressionHead

    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 numerical_feature_embedding: layers.Layer = None,
                 saint_encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = RegressionHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         numerical_feature_embedding=numerical_feature_embedding,
                         saint_encoder=saint_encoder,
                         head=head,
                         **kwargs)


class SAINTPretrainer(_BaseSAINTPretrainer):
    """
    SAINTPretrainer model with LayerFlow design.
    It is based on the pretraining architecture of the SAINT model
    proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        model: `keras.Model`,
            An instance of the SAINT model that is to be pretrained.
        mixup: `layers.Layer`,
            An instance of `MixUp` layer or any custom layer that can work
            in its place.
            You can import the `MixUp` layer as follows,
                >>> from teras.layerflow.layers import MixUp

        cutmix: `layers.Layer`,
            An instance of `CutMix` layer or any custom layer that can work
            in its place.
            You can import the `CutMix` layer as follows,
                >>> from teras.layerflow.layers import CutMix

        projection_head_1: `layers.Layer`,
            An instance of `ProjectionHead` layer that is used to project embeddings
            of *real* samples to a lower dimensionality before reconstructing the inputs.
            You can import the `ProjectionHead` layer as follows,
                >>> from teras.layerflow.layers.saint import ProjectionHead

        projection_head_2: `layers.Layer`,
            An instance of `ProjectionHead` layer that is used to project embeddings
            of *augmented* samples to a lower dimensionality before reconstructing the inputs.
            You can import the `ProjectionHead` layer as follows,
                >>> from teras.layerflow.layers.saint import ProjectionHead

        reconstruction_head: `layers.Layer`,
            An instance of `ReconstructionHead` which applies a separate ReconstructionHeadBlock
            to reconstruct the input features.
                >>> from teras.layerflow.layers.saint import ReconstructionHead

        temperature: `float`, default 0.7,
            Temperature value used in the computation of the InfoNCE contrastive loss.
        lambda_: `float`, default 10,
            Controls the weightage of denoising loss in the summation of denoising and
            contrastive loss.
    """
    def __init__(self,
                 model: SAINT,
                 mixup: layers.Layer = None,
                 cutmix: layers.Layer = None,
                 projection_head_1: layers.Layer = None,
                 projection_head_2: layers.Layer = None,
                 reconstruction_head: layers.Layer = None,
                 temperature: float = 0.7,
                 lambda_: float = 10.,
                 **kwargs):
        super().__init__(model=model,
                         temperature=temperature,
                         lambda_=lambda_,
                         **kwargs)
        if mixup is not None:
            self.mixup = mixup

        if cutmix is not None:
            self.cutmix = cutmix

        if projection_head_1 is not None:
            self.projection_head_1 = projection_head_1

        if projection_head_2 is not None:
            self.projection_head_2 = projection_head_2

        if reconstruction_head is not None:
            self.reconstruction_head = reconstruction_head
