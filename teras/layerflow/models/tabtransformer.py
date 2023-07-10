from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.tabtransformer import (ClassificationHead,
                                         RegressionHead)
from teras.models import (TabTransformer as BaseTabTransformer,
                          TabTransformerPretrainer)
from teras.layerflow.models import SimpleModel


class TabTransformer(BaseTabTransformer):
    """
    Base TabTransformer model class with LayerFlow design.

    TabTransformer architecture is proposed by Xin Huang et al.
    in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer is a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

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

        column_embedding: `layers.Layer`,
            An instance of `TabTransformerColumnEmbedding` layer to apply over categorical embeddings,
            or any layer that can work in place of `TabTransformerColumnEmbedding` for that purpose.
            If None, a `TabTransformerColumnEmbedding` layer with default values will be used.
            You can import the `TabTransformerColumnEmbedding` layer as follows,
                >>> from teras.layerflow.layers import TabTColumnEmbedding

        encoder: `layers.Layer`,
            An instance of Encoder layer to encode feature embeddings,
            or any layer that can work in place of Encoder for that purpose.
            If None, an Encoder layer with default values will be used.
                >>> from teras.layerflow.layers import Encoder

        head: `layers.Layer`,
            An instance of ClassificationHead or RegressionHead layer for final outputs,
            or any layer that can work in place of a Head layer for that purpose.
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 column_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         **kwargs)
        if categorical_feature_embedding is not None:
            self.categorical_feature_embedding = categorical_feature_embedding

        if column_embedding is not None:
            self.column_embedding = column_embedding

        if encoder is not None:
            self.encoder = encoder

        self.head = head

    def get_config(self):
        config = super().get_config()
        new_config = {'categorical_feature_embedding': keras.layers.serialize(self.categorical_feature_embedding),
                      'column_embedding': keras.layers.serialize(self.column_embedding),
                      'encoder': keras.layers.serialize(self.encoder),
                      'head': keras.layers.serialize(self.head),
                      }
        config.update(new_config)
        return config


class TabTransformerClassifier(TabTransformer):
    """
    TabTransformerClassifier model class with LayerFlow design.

    TabTransformer architecture is proposed by Xin Huang et al.
    in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer is a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

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

        column_embedding: `layers.Layer`,
            An instance of `TabTransformerColumnEmbedding` layer to apply over categorical embeddings,
            or any layer that can work in place of `TabTransformerColumnEmbedding` for that purpose.
            If None, a `TabTransformerColumnEmbedding` layer with default values will be used.
            You can import the `TabTransformerColumnEmbedding` layer as follows,
                >>> from teras.layerflow.layers import TabTColumnEmbedding

        encoder: `layers.Layer`,
            An instance of Encoder layer to encode feature embeddings,
            or any layer that can work in place of Encoder for that purpose.
            If None, an Encoder layer with default values will be used.
                >>> from teras.layerflow.layers import Encoder

        head: `layers.Layer`,
            An instance of `TabTClassificationHead` layer for the final outputs,
            or any layer that can work in place of a `TabTClassificationHead` layer for that purpose.
            If None, `TabTClassificationHead` layer with default values will be used.
            You can import the `TabTClassificationHead` as follows,
                >>> from teras.layerflow.layers import TabTClassificationHead
    """

    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 column_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = ClassificationHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         column_embedding=column_embedding,
                         encoder=encoder,
                         head=head,
                         **kwargs)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model: TabTransformer,
                        head: layers.Layer = None
                        ):
        """
        Class method to create a TabTransformer Classifier model instance from
        a pretrained base TabTransformer model instance.

        Args:
            pretrained_model: `TabTransformer`,
                A pretrained base TabTransformer model instance.
           head: `layers.Layer`,
                An instance of `TabTClassificationHead` layer for the final outputs,
                or any layer that can work in place of a `TabTClassificationHead` layer for that purpose.
                If None, `TabTClassificationHead` layer with default values will be used.
                You can import `TabTClassificationHead` as follows,
                    >>> from teras.layerflow.layers import TabTClassificationHead

        Returns:
            A TabTransformer Classifier instance based of the pretrained model.
        """
        if head is None:
            head = ClassificationHead(name="tabtransformer_classification_head")
        model = SimpleModel(body=pretrained_model,
                            head=head,
                            name="tabtransformer_classifier_pretrained")
        return model


class TabTransformerRegressor(TabTransformer):
    """
    TabTransformerRegressor model class with LayerFlow design.

    TabTransformer architecture is proposed by Xin Huang et al.
    in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer is a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

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

        column_embedding: `layers.Layer`,
            An instance of `TabTransformerColumnEmbedding` layer to apply over categorical embeddings,
            or any layer that can work in place of `TabTransformerColumnEmbedding` for that purpose.
            If None, a `TabTransformerColumnEmbedding` layer with default values will be used.
            You can import the `TabTransformerColumnEmbedding` layer as follows,
                >>> from teras.layerflow.layers import TabTColumnEmbedding

        encoder: `layers.Layer`,
            An instance of Encoder layer to encode feature embeddings,
            or any layer that can work in place of Encoder for that purpose.
            If None, an Encoder layer with default values will be used.
                >>> from teras.layerflow.layers import Encoder

        head: `layers.Layer`,
            An instance of `TabTRegressionHead` layer for the final outputs,
            or any layer that can work in place of a `TabTRegressionHead` layer for that purpose.
            If None, `TabTRegressionHead` layer with default values will be used.
            You can import `TabTRegressionHead` as follows,
                >>> from teras.layerflow.layers import TabTRegressionHead
    """
    def __init__(self,
                 features_metadata: dict,
                 categorical_feature_embedding: layers.Layer = None,
                 column_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = RegressionHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         column_embedding=column_embedding,
                         encoder=encoder,
                         head=head,
                         **kwargs)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model: TabTransformer,
                        head: layers.Layer = None,
                        ):
        """
        Class method to create a TabTransformer Regressor model instance from
        a pretrained base TabTransformer model instance.

        Args:
            pretrained_model: `TabTransformer`,
                A pretrained base TabTransformer model instance.
            head: `layers.Layer`,
                An instance of RegressionHead layer for the final outputs,
                or any layer that can work in place of a RegressionHead layer for that purpose.
                If None, RegressionHead layer with default values will be used.
                You can import `TabTRegressionHead` as follows,
                    >>> from teras.layerflow.layers import TabTRegressionHead

        Returns:
            A TabTransformer Regressor instance based of the pretrained model.
        """
        if head is None:
            head = RegressionHead(name="tabtransformer_regression_head")
        model = SimpleModel(body=pretrained_model,
                            head=head,
                            name="tabtransformer_regressor_pretrained")
        return model
