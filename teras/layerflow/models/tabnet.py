from tensorflow.keras import layers, models
from teras.models.tabnet import TabNet as _BaseTabNet
from teras.layerflow.layers.tabnet import ClassificationHead, RegressionHead


class TabNet(_BaseTabNet):
    """
    Base TabNet model class with LayerFlow design.

    TabNet is a novel high-performance and interpretable canonical
    deep tabular data learning architecture.
    TabNet uses sequential attention to choose which features to reason
    from at each decision step, enabling interpretability and more
    efficient learning as the learning capacity is used for the most
    salient features.

    TabNet is proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

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
                 categorical_features_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         **kwargs)
        if categorical_features_embedding is not None:
            self.categorical_features_embedding = categorical_features_embedding

        if encoder is not None:
            self.encoder = encoder

        if head is not None:
            self.head = head


class TabNetClassifier(TabNet):
    """
    TabNetClassifier with LayerFlow desing.
    It is based on the TabNet archietcture.

    TabNet is a novel high-performance and interpretable canonical
    deep tabular data learning architecture.
    TabNet uses sequential attention to choose which features to reason
    from at each decision step, enabling interpretability and more
    efficient learning as the learning capacity is used for the most
    salient features.

    TabNet is proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

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
                 categorical_features_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = ClassificationHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_features_embedding=categorical_features_embedding,
                         encoder=encoder,
                         head=head,
                         **kwargs)


class TabNetRegressor(TabNet):
    """
    TabNetRegressor with LayerFlow desing.
    It is based on the TabNet archietcture.

    TabNet is a novel high-performance and interpretable canonical
    deep tabular data learning architecture.
    TabNet uses sequential attention to choose which features to reason
    from at each decision step, enabling interpretability and more
    efficient learning as the learning capacity is used for the most
    salient features.

    TabNet is proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

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
                 categorical_features_embedding: layers.Layer = None,
                 encoder: layers.Layer = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = RegressionHead()
        super().__init__(features_metadata=features_metadata,
                         categorical_features_embedding=categorical_features_embedding,
                         encoder=encoder,
                         head=head,
                         **kwargs)
