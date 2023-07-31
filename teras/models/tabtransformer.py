from tensorflow import keras
from teras.config.tabtransformer import TabTransformerConfig
from teras.layerflow.models.tabtransformer import TabTransformer as TabTransformerLF
from teras.layers.common.transformer import Encoder
from teras.layers.categorical_feature_embedding import CategoricalFeatureEmbedding
from teras.layers.numerical_feature_normalization import NumericalFeatureNormalization
from teras.layers.tabtransformer.tabtransformer_column_embedding import TabTransformerColumnEmbedding
from teras.layers.common.head import ClassificationHead, RegressionHead
from teras.utils.types import UnitsValuesType, NormalizationType


@keras.saving.register_keras_serializable("keras.models")
class TabTransformer(TabTransformerLF):
    """
    TabTransformer architecture as proposed by Xin Huang et al.
    in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer, a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Dimensionality of the learnable feature embeddings for categorical features.

        num_transformer_layers: ``int``, default 6,
            Number of transformer layers to use in the ``Encoder`` layer.
            The ``Encoder`` is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the ``Transformer`` layer which in turn is part of the
            ``Encoder`` layer.

        attention_dropout: ``float``, default 0.0,
            Dropout rate to use in the ``MultiHeadSelfAttention`` layer in the transformer layer.

        feedforward_dropout: ``float``, default 0.0,
            Dropout rate to use for the ``Dropout`` layer in the ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the ``embedding_dim``
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the ``FeedForward`` layer.

        norm_epsilon: ``float``, default 1e-6,
            A very small number used for normalization in the ``LayerNormalization`` layer.

        use_column_embedding: ``bool``, default True,
            Whether to use the novel ``ColumnEmbedding`` layer proposed in the
            ``TabTransformer`` architecture for the categorical features.
            The ``ColumnEmbedding`` layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.

        numerical_normalization: ``str`` or ``keras.layes.Layer``, default "layer",
            Normalization to use for the numerical features.
"""
    def __init__(self,
                 input_dim: int,
                 features_metadata: dict,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 feedforward_multiplier: int = TabTransformerConfig.feedforward_multiplier,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 numerical_normalization: NormalizationType = TabTransformerConfig.numerical_normalization,
                 **kwargs
                 ):

        if features_metadata is None:
            raise ValueError(f"""
            `features_metadata` is required for embedding features and hence cannot be None.            
            You can get this features_metadata dictionary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features, numerical_features)`
            Received, `features_metadata`: {features_metadata}
            """)

        categorical_feature_embedding = CategoricalFeatureEmbedding(
            features_metadata=features_metadata,
            embedding_dim=embedding_dim,
            encode=encode_categorical_values
        )
        column_embedding = TabTransformerColumnEmbedding(embedding_dim=embedding_dim,
                                                         num_categorical_features=len(features_metadata["categorical"]))
        encoder = Encoder(num_transformer_layers=num_transformer_layers,
                          num_attention_heads=num_attention_heads,
                          embedding_dim=embedding_dim,
                          attention_dropout=attention_dropout,
                          feedforward_dropout=feedforward_dropout,
                          feedforward_multiplier=feedforward_multiplier,
                          norm_epsilon=norm_epsilon)
        numerical_feature_normalization = NumericalFeatureNormalization(features_metadata=features_metadata,
                                                                        normalization=numerical_normalization)

        super().__init__(input_dim=input_dim,
                         categorical_feature_embedding=categorical_feature_embedding,
                         column_embedding=column_embedding,
                         encoder=encoder,
                         numerical_feature_normalization=numerical_feature_normalization,
                         **kwargs)

        self.input_dim = input_dim
        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.use_column_embedding = use_column_embedding
        self.norm_epsilon = norm_epsilon
        self.encode_categorical_values = encode_categorical_values
        self.numerical_normalization = numerical_normalization

    def get_config(self):
        numerical_normalization_serialized = self.numerical_normalization
        if not isinstance(numerical_normalization_serialized, str):
            numerical_normalization_serialized = keras.layers.serialize(numerical_normalization_serialized)
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'features_metadata': self.features_metadata,
                  'embedding_dim': self.embedding_dim,
                  'num_transformer_layers': self.num_transformer_layers,
                  'num_attention_heads': self.num_attention_heads,
                  'attention_dropout': self.attention_dropout,
                  'feedforward_dropout': self.feedforward_dropout,
                  'feedforward_multiplier': self.feedforward_multiplier,
                  'norm_epsilon': self.norm_epsilon,
                  'use_column_embedding': self.use_column_embedding,
                  'encode_categorical_values': self.encode_categorical_values,
                  'numerical_normalization': numerical_normalization_serialized,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        # Super class i.e. TabTransformerLF's ``from_config`` method  would try
        # to build the TabTransformerLF class expecting configuration to contain
        # sublayers' configs such as ``CategoricalFeatureEmebedding`` layer's config etc.
        # Hence, overriding the from_config method here is imperative.
        #
        # !NOTE: Since in this default/parametric API version, we don't take any
        # sublayers as arguments, so, we don't need to deserialize anything.
        input_dim = config.pop("input_dim")
        features_metadata = config.pop("features_metadata")
        return cls(input_dim=input_dim,
                   features_metadata=features_metadata,
                   **config)


class TabTransformerClassifier(TabTransformer):
    """
    TabTransformerClassifier based on the TabTransformer architecture
    as proposed by Xin Huang et al. in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer, a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        num_classes: ``int``, default 2,
            Number of classes to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default [64, 32],
            Units values to use in the hidden layers in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Dimensionality of the learnable feature embeddings for categorical features.

        num_transformer_layers: ``int``, default 6,
            Number of transformer layers to use in the ``Encoder`` layer.
            The ``Encoder`` is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the ``Transformer`` layer which in turn is part of the
            ``Encoder`` layer.

        attention_dropout: ``float``, default 0.0,
            Dropout rate to use in the ``MultiHeadSelfAttention`` layer in the transformer layer.

        feedforward_dropout: ``float``, default 0.0,
            Dropout rate to use for the ``Dropout`` layer in the ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the ``embedding_dim``
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the ``FeedForward`` layer.

        norm_epsilon: ``float``, default 1e-6,
            A very small number used for normalization in the ``LayerNormalization`` layer.

        use_column_embedding: ``bool``, default True,
            Whether to use the novel ``ColumnEmbedding`` layer proposed in the
            ``TabTransformer`` architecture for the categorical features.
            The ``ColumnEmbedding`` layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.

        numerical_normalization: ``str`` or ``keras.layes.Layer``, default "layer",
            Normalization to use for the numerical features.
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: UnitsValuesType = (64, 32),
                 input_dim: int = None,
                 features_metadata: dict = None,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 feedforward_multiplier: int = TabTransformerConfig.feedforward_multiplier,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 numerical_normalization: NormalizationType = TabTransformerConfig.numerical_normalization,
                 **kwargs
                 ):
        # Since the parent `TabTransformer` class subclasses its LayerFlow version
        # which accepts layers as input and does expose a head argumnet to pass the layer.
        # we can create a relevant head layer, in this case, the `ClassificationHead` layer
        # and pass its instance to the parent who will pass it along to its parent as part of
        # the **kwargs dict
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_values,
                                  activation_hidden="relu",
                                  normalization="batch",
                                  name="tabtransformer_classifier_head")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         norm_epsilon=norm_epsilon,
                         use_column_embedding=use_column_embedding,
                         encode_categorical_values=encode_categorical_values,
                         numerical_normalization=numerical_normalization,
                         head=head,
                         **kwargs)
        self.num_classes = num_classes
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'head_units_values': self.head_units_values
                       })
        return config


class TabTransformerRegressor(TabTransformer):
    """
    TabTransformerRegressor based on the TabTransformer architecture
    as proposed by Xin Huang et al. in the paper,
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    TabTransformer, a novel deep tabular data modeling architecture for
    supervised and semi-supervised learning.
    The TabTransformer is built upon self-attention based Transformers.
    The Transformer layers transform the embeddings of categorical features
    into robust contextual embeddings to achieve higher prediction accuracy.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        num_outputs: ``int``, default 1,
            Number of regression outputs to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default [64, 32],
            Units values to use in the hidden layers in the ``TabTRegressionHead``.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Dimensionality of the learnable feature embeddings for categorical features.

        num_transformer_layers: ``int``, default 6,
            Number of transformer layers to use in the ``Encoder`` layer.
            The ``Encoder`` is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the ``Transformer`` layer which in turn is part of the
            ``Encoder`` layer.

        attention_dropout: ``float``, default 0.0,
            Dropout rate to use in the ``MultiHeadSelfAttention`` layer in the transformer layer.

        feedforward_dropout: ``float``, default 0.0,
            Dropout rate to use for the ``Dropout`` layer in the ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the ``embedding_dim``
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the ``FeedForward`` layer.

        norm_epsilon: ``float``, default 1e-6,
            A very small number used for normalization in the ``LayerNormalization`` layer.

        use_column_embedding: ``bool``, default True,
            Whether to use the novel ``ColumnEmbedding`` layer proposed in the
            ``TabTransformer`` architecture for the categorical features.
            The ``ColumnEmbedding`` layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.

        numerical_normalization: ``str`` or ``keras.layes.Layer``, default "layer",
            Normalization to use for the numerical features.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 head_units_values: UnitsValuesType = (64, 32),
                 input_dim: int = None,
                 features_metadata: dict = None,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 feedforward_multiplier: int = TabTransformerConfig.feedforward_multiplier,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 numerical_normalization: NormalizationType = TabTransformerConfig.numerical_normalization,
                 **kwargs
                 ):
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_values,
                              activation_hidden="relu",
                              normalization="batch",
                              name="tabtransformer_regressor_head")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         norm_epsilon=norm_epsilon,
                         use_column_embedding=use_column_embedding,
                         encode_categorical_values=encode_categorical_values,
                         numerical_normalization=numerical_normalization,
                         head=head,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_outputs': self.num_outputs,
                       'head_units_values': self.head_units_values,
                       })
        return config
