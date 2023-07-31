from tensorflow import keras
from teras.layers.ft_transformer.ft_numerical_feature_embedding import FTNumericalFeatureEmbedding
from teras.layers.ft_transformer.ft_cls_token import FTCLSToken
from teras.layers.categorical_feature_embedding import CategoricalFeatureEmbedding
from teras.layers.common.transformer import Encoder
from teras.layers.common.head import ClassificationHead, RegressionHead
from teras.layerflow.models.ft_transformer import FTTransformer as _FTTransformerLF
from teras.utils.types import UnitsValuesType


@keras.saving.register_keras_serializable("teras.models")
class FTTransformer(_FTTransformerLF):
    """
    FT Transformer architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

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
            Embedding dimensions used in embedding numerical and categorical features.

        numerical_embedding_hidden_dim: ``int`` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            SAINT NumericalFeatureEmebedding layer.

        num_transformer_layer: ``int``, default 6,
            Number of ``Transformer`` layers to use in the ``Encoder``.
            The encoder is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.

        attention_dropout: ``float``, default 0.1,
            Dropout rate to use in the ``MultiHeadSelfAttention`` layer in the transformer layer.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate to use for the dropout layer in the FeedForward block.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multipled with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.
    """
    def __init__(self,
                 input_dim: int,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 8,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 feedforward_dropout:  float = 0.05,
                 feedforward_multiplier: int = 4,
                 encode_categorical_values: bool = True,
                 **kwargs):
        num_categorical_features = len(features_metadata["categorical"])
        num_numerical_features = len(features_metadata["numerical"])
        categorical_features_exist = num_categorical_features > 0
        numerical_features_exist = num_numerical_features > 0

        # Numerical/Continuous Features Embedding
        numerical_feature_embedding = None
        if numerical_features_exist:
            numerical_feature_embedding = FTNumericalFeatureEmbedding(features_metadata=features_metadata,
                                                                      embedding_dim=embedding_dim)

        # Categorical Features Embedding
        categorical_feature_embedding = None
        if categorical_features_exist:
            # If categorical features exist, then they must be embedded
            categorical_feature_embedding = CategoricalFeatureEmbedding(
                features_metadata=features_metadata,
                embedding_dim=embedding_dim,
                encode=encode_categorical_values)

        cls_token = FTCLSToken(embedding_dim,
                               initialization="normal")
        encoder = Encoder(num_transformer_layers=num_transformer_layers,
                          num_attention_heads=num_attention_heads,
                          embedding_dim=embedding_dim,
                          attention_dropout=attention_dropout,
                          feedforward_dropout=feedforward_dropout,
                          feedforward_multiplier=feedforward_multiplier)

        super().__init__(input_dim=input_dim,
                         categorical_feature_embedding=categorical_feature_embedding,
                         numerical_feature_embedding=numerical_feature_embedding,
                         cls_token=cls_token,
                         encoder=encoder,
                         **kwargs
                         )
        self.input_dim = input_dim
        self.features_metadata = features_metadata
        self.num_transformer_layers = num_transformer_layers
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.encode_categorical_values = encode_categorical_values

    def get_config(self):
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
                  'encode_categorical_values': self.encode_categorical_values}
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        features_metadata = config.pop("features_metadata")
        return cls(input_dim=input_dim,
                   features_metadata=features_metadata,
                   **config)


@keras.saving.register_keras_serializable("teras.models")
class FTTransformerClassifier(FTTransformer):
    """
    FTTransformerClassifier based on the FTTransformer architecture proposed
    by Yury Gorishniy et al. in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: ``int``, default 2,
            Number of classes to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Units values to use in the hidden layers in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

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
            Embedding dimensions used in embedding numerical and categorical features.

        numerical_embedding_hidden_dim: ``int`` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            SAINT NumericalFeatureEmebedding layer.

        num_transformer_layer: ``int``, default 6,
            Number of ``Transformer`` layers to use in the ``Encoder``.
            The encoder is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.

        attention_dropout: ``float``, default 0.1,
            Dropout rate to use in the ``MultiHeadSelfAttention`` layer in the transformer layer.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate to use for the dropout layer in the FeedForward block.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multipled with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_value: UnitsValuesType = None,
                 features_metadata: dict = None,
                 input_dim: int = None,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 8,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 feedforward_dropout:  float = 0.05,
                 feedforward_multiplier: int = 4,
                 encode_categorical_values: bool = True,
                 **kwargs):
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_value,
                                  normalization="layer")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         encode_categorical_values=encode_categorical_values,
                         head=head,
                         **kwargs)
        self.num_classes = num_classes
        self.head_units_value = head_units_value

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'head_units_value': self.head_units_value})
        return config


@keras.saving.register_keras_serializable("teras.models")
class FTTransformerRegressor(FTTransformer):
    """
    FTTransformerRegressor based on the FTTransformer architecture proposed
    by Yury Gorishniy et al. in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_outputs: `int`, default 1,
            Number of outputs to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Units values to use in the hidden layers in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

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
            Embedding dimensions used in embedding numerical and categorical features.

        numerical_embedding_hidden_dim: ``int`` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            SAINT NumericalFeatureEmebedding layer.

        num_transformer_layer: ``int``, default 6,
            Number of ``Transformer`` layers to use in the ``Encoder``.
            The encoder is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.

        attention_dropout: ``float``, default 0.1,
            Dropout rate to use in the ``MultiHeadSelfAttention`` layer in the transformer layer.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate to use for the dropout layer in the FeedForward block.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multipled with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.
    """

    def __init__(self,
                 num_outputs: int = 1,
                 head_units_value: UnitsValuesType = None,
                 input_dim: int = None,
                 features_metadata: dict = None,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 8,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 feedforward_dropout: float = 0.05,
                 feedforward_multiplier: int = 4,
                 encode_categorical_values: bool = True,
                 **kwargs):
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_value,
                              normalization="layer")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         encode_categorical_values=encode_categorical_values,
                         head=head,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_value = head_units_value

    def get_config(self):
        config = super().get_config()
        config.update({'num_outputs': self.num_outputs,
                       'head_units_value': self.head_units_value})
        return config
