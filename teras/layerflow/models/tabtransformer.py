from tensorflow.keras import layers, models
from teras.layers.tabtransformer import (ClassificationHead,
                                         RegressionHead)
from teras.models import (TabTransformer as BaseTabTransformer,
                          TabTransformerPretrainer)


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
        categorical_features_embedding: `layers.Layer`,
            An instance of CategoricalFeatureEmbedding layer to embedd categorical features
            or any layer that can work in place of CategoricalFeatureEmbedding for that purpose.
            If None, a CategoricalFeatureEmbedding layer with default values will be used.
        column_embedding: `layers.Layer`,
            An instance of ColumnEmbedding layer to apply over categorical embeddings,
            or any layer that can work in place of ColumnEmbedding for that purpose.
            If None, a ColumnEmbedding layer with default values will be used.
        encoder: `layers.Layer`,
            An instance of Encoder layer to encode feature embeddings,
            or any layer that can work in place of Encoder for that purpose.
            If None, an Encoder layer with default values will be used.
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
        categorical_features_embedding: `layers.Layer`,
            An instance of CategoricalFeatureEmbedding layer to embedd categorical features
            or any layer that can work in place of CategoricalFeatureEmbedding for that purpose.
            If None, a CategoricalFeatureEmbedding layer with default values will be used.
        column_embedding: `layers.Layer`,
            An instance of ColumnEmbedding layer to apply over categorical embeddings,
            or any layer that can work in place of ColumnEmbedding for that purpose.
            If None, a ColumnEmbedding layer with default values will be used.
        encoder: `layers.Layer`,
            An instance of Encoder layer to encode feature embeddings,
            or any layer that can work in place of Encoder for that purpose.
            If None, an Encoder layer with default values will be used.
        head: `layers.Layer`,
            An instance of ClassificationHead layer for the final outputs,
            or any layer that can work in place of a ClassificationHead layer for that purpose.
            If None, ClassificationHead layer with default values will be used.
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
        categorical_features_embedding: `layers.Layer`,
            An instance of CategoricalFeatureEmbedding layer to embedd categorical features
            or any layer that can work in place of CategoricalFeatureEmbedding for that purpose.
            If None, a CategoricalFeatureEmbedding layer with default values will be used.
        column_embedding: `layers.Layer`,
            An instance of ColumnEmbedding layer to apply over categorical embeddings,
            or any layer that can work in place of ColumnEmbedding for that purpose.
            If None, a ColumnEmbedding layer with default values will be used.
        encoder: `layers.Layer`,
            An instance of Encoder layer to encode feature embeddings,
            or any layer that can work in place of Encoder for that purpose.
            If None, an Encoder layer with default values will be used.
        head: `layers.Layer`,
            An instance of RegressionHead layer for the final outputs,
            or any layer that can work in place of a RegressionHead layer for that purpose.
            If None, RegressionHead layer with default values will be used.
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
