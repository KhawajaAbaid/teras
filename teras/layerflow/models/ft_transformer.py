from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layerflow.layers.ft_transformer import (ClassificationHead,
                                                   RegressionHead)
from teras.models import (FTTransformer as _BaseFTTransformer,
                          FTTransformerClassifier as _BaseFTTransformerClassifier,
                          FTTransformerRegressor as _BaseFTTransformerRegressor)


class FTTransformer(_BaseFTTransformer):
    """
    FTTransformer architecture with LayrFlow design.
    FT-Transformer is proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

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
            An instance of `FTNumericalFeatureEmbedding` layer to embedd numerical features
            or any layer that can work in place of `FTNumericalFeatureEmbedding` for that purpose.
            If None, a `FTNumericalFeatureEmbedding` layer with default values will be used.
            You can import the `FTNumericalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import FTNumericalFeatureEmbedding

        encoder: `layers.Layer`,
            An instance of `Encoder` layer to encode the features embeddings,
            or any layer that can work in place of `Encoder` for that purpose.
            If None, a `Encoder` layer with default values will be used.
            You can import the `Encoder` layer as follows,
                >>> from teras.layerflow.layers import Encoder

        head: `layers.Layer`,
            An instance of FTClassificationHead or FTRegressionHead layer for final outputs,
            or any layer that can work in place of a Head layer for that purpose.
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 numerical_feature_embedding: layers.Layer = None,
                 cls_token: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         **kwargs)
        if categorical_feature_embedding is not None:
            self.categorical_feature_embedding = categorical_feature_embedding

        if numerical_feature_embedding is not None:
            self.numerical_feature_embedding = numerical_feature_embedding

        if cls_token is not None:
            self.cls_token = cls_token

        if encoder is not None:
            self.encoder = encoder

        if head is not None:
            self.head = head

    def get_config(self):
        config = super().get_config()
        new_config = {'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                      'numerical_feature_embedding': keras.layers.serialize(self.numerical_feature_embedding),
                      'cls_token': keras.layers.serialize(self.cls_token),
                      'encoder': keras.layers.serialize(self.encoder),
                      'head': keras.layers.serialize(self.head),
                      }
        config.update(new_config)
        return config


class FTTransformerClassifier(_BaseFTTransformerClassifier):
    """
    FTTransformerClassifier architecture with LayrFlow design.
    It is based on the FT-Transformer architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

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
            An instance of `FTNumericalFeatureEmbedding` layer to embedd numerical features
            or any layer that can work in place of `FTNumericalFeatureEmbedding` for that purpose.
            If None, a `FTNumericalFeatureEmbedding` layer with default values will be used.
            You can import the `FTNumericalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import FTNumericalFeatureEmbedding

        encoder: `layers.Layer`,
            An instance of `Encoder` layer to encode the features embeddings,
            or any layer that can work in place of `Encoder` for that purpose.
            If None, a `Encoder` layer with default values will be used.
            You can import the `Encoder` layer as follows,
                >>> from teras.layerflow.layers import Encoder

        head: `layers.Layer`,
            An instance of `FTClassificationHead` layer for the final outputs,
            or any layer that can work in place of a `FTClassificationHead` layer for that purpose.
            If None, `FTClassificationHead` layer with default values will be used.
            You can import the `FTClassificationHead` layer as follows,
                >>> from teras.layerflow.layers import FTClassificationHead
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 numerical_feature_embedding: layers.Layer = None,
                 cls_token: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = ClassificationHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         numerical_feature_embedding=numerical_feature_embedding,
                         cls_token=cls_token,
                         encoder=encoder,
                         head=head,
                         **kwargs)


class FTTransformerRegressor(_BaseFTTransformerRegressor):
    """
    FTTransformerRegressor architecture with LayrFlow design.
    It is based on the FT-Transformer architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

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
            An instance of `FTNumericalFeatureEmbedding` layer to embedd numerical features
            or any layer that can work in place of `FTNumericalFeatureEmbedding` for that purpose.
            If None, a `FTNumericalFeatureEmbedding` layer with default values will be used.
            You can import the `FTNumericalFeatureEmbedding` layer as follows,
                >>> from teras.layerflow.layers import FTNumericalFeatureEmbedding

        encoder: `layers.Layer`,
            An instance of `Encoder` layer to encode the features embeddings,
            or any layer that can work in place of `Encoder` for that purpose.
            If None, a `Encoder` layer with default values will be used.
            You can import the `Encoder` layer as follows,
                >>> from teras.layerflow.layers import Encoder

        head: `layers.Layer`,
            An instance of `FTRegressionHead` layer for the final outputs,
            or any layer that can work in place of a `FTRegressionHead` layer for that purpose.
            If None, `FTRegressionHead` layer with default values will be used.
            You can import the `FTRegressionHead` layer as follows,
                >>> from teras.layerflow.layers import FTRegressionHead
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 numerical_feature_embedding: layers.Layer = None,
                 cls_token: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = RegressionHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         numerical_feature_embedding=numerical_feature_embedding,
                         cls_token=cls_token,
                         encoder=encoder,
                         head=head,
                         **kwargs)
