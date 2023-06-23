import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from warnings import warn
from teras.layers import CategoricalFeatureEmbedding
from teras.layers import SAINTNumericalFeatureEmbedding, SAINTEncoder
from teras.layers.common.transformer import (ClassificationHead,
                                             RegressionHead)
from teras.config.saint import SAINTConfig
from typing import Union, List, Tuple


FEATURE_NAMES_TYPE = Union[List[str], Tuple[str]]
UNITS_VALUES_TYPE = Union[List[int], Tuple[int]]


class SAINT(keras.Model):
    """
    SAINT architecture proposed by Gowthami Somepalli et al.
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
        embedding_dim: `int`, default 32,
            Embedding dimensions used in embedding numerical and categorical features.
        numerical_embedding_hidden_dim: `int` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            SAINT NumericalFeatureEmebedding layer.
        num_transformer_layer: `int`, default 6,
            Number of (SAINT) transformer layers to use in the Encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8,
            Number of attention heads to use in the MultiHeadSelfAttention layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.
        num_inter_sample_attention_heads: `int`, default 8,
            Number of heads to use in the MultiHeadInterSampleAttention that applies
            attention over rows.
        attention_dropout: `float`, default 0.1, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        inter_sample_attention_dropout: `float`, default 0.1,
            Dropout rate for MultiHeadInterSampleAttention layer that applies
            attention over rows.
        feedforward_dropout: `float`, default 0.1,
            Dropout rate to use for the dropout layer in the FeedForward block.
        norm_epsilon: `float`, default 1e-6,
            A very small number used for normalization in the `LayerNormalization` layer.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
        embed_numerical_features: `bool`, default True,
            Whether to embed the numerical features.
            If False, (SAINT) `NumericalFeatureEmbedding` layer won't be applied to
            numerical features instead they will just be normalized using `LayerNormaliztion`.
        apply_attention_to_features: `bool`, default True,
            Whether to apply attention over features using the regular `MultiHeadAttenion` layer.
        apply_attention_to_rows: `bool`, default True,
            Whether to apply attention over rows using the SAINT `MultiHeadInterSampleAttention`
            layer.
            Although it is strongly recommended to apply attention to both rows and features,
            but for experimentation's sake you can disable one of them, but NOT both at the
            same time!
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = SAINTConfig.embedding_dim,
                 numerical_embedding_hidden_dim: int = SAINTConfig.numerical_embedding_hidden_dim,
                 num_transformer_layers: int = SAINTConfig.num_transformer_layers,
                 num_attention_heads: int = SAINTConfig.num_attention_heads,
                 num_inter_sample_attention_heads: int = SAINTConfig.num_inter_sample_attention_heads,
                 attention_dropout: float = SAINTConfig.attention_dropout,
                 inter_sample_attention_dropout: float = SAINTConfig.inter_sample_attention_dropout,
                 feedforward_dropout: float = SAINTConfig.feedforward_dropout,
                 norm_epsilon: float = SAINTConfig.norm_epsilon,
                 encode_categorical_values: bool = SAINTConfig.encode_categorical_values,
                 embed_numerical_features: bool = SAINTConfig.embed_numerical_features,
                 apply_attention_to_features: bool = SAINTConfig.apply_attention_to_features,
                 apply_attention_to_rows: bool = SAINTConfig.apply_attention_to_rows,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.num_inter_sample_attention_heads = num_inter_sample_attention_heads
        self.attention_dropout = attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.encode_categorical_values = encode_categorical_values
        self.embed_numerical_features = embed_numerical_features
        self.numerical_embedding_hidden_dim = numerical_embedding_hidden_dim
        self.apply_attention_to_features = apply_attention_to_features
        self.apply_attention_to_rows = apply_attention_to_rows

        self._categorical_features_metadata = self.features_metadata["categorical"]
        self._numerical_features_metadata = self.features_metadata["numerical"]
        self._num_categorical_features = len(self._categorical_features_metadata)
        self._num_numerical_features = len(self._numerical_features_metadata)

        self._num_embedded_features = 0

        self._numerical_features_exists = self._num_numerical_features > 0
        self._categorical_features_exist = self._num_categorical_features > 0

        # Numerical/Continuous Features Embedding
        self.numerical_feature_embedding = None
        if self.embed_numerical_features:
            if self._numerical_features_exists:
                self.numerical_feature_embedding = SAINTNumericalFeatureEmbedding(
                    embedding_dim=self.embedding_dim,
                    hidden_dim=self.numerical_embedding_hidden_dim,
                    numerical_features_metadata=self._numerical_features_metadata
                )
                self._num_embedded_features += self._num_numerical_features
            else:
                # Numerical features don't exist
                warn("`embed_numerical_features` is set to True, but no numerical features exist in the "
                     "`features_metadata` dictionary, hence it is assumed that no numerical features exist "
                     "in the given dataset. "
                     "But if numerical features do exist in the dataset, then make sure to pass them when "
                     "you call the `get_features_metadata_for_embedding` function. And train this this model again. ")
        else:
            # embed_numerical_features is set to False by the user.
            if self._numerical_features_exists:
                # But numerical features exist, so warn the user
                warn("`embed_numerical_features` is set to False but numerical features exist in the dataset. "
                     "It is recommended to embed the numerical features for better performance. ")

        # Categorical Features Embedding
        self.categorical_feature_embedding = None
        if self._categorical_features_exist:
            # If categorical features exist, then they must be embedded
            self.categorical_feature_embedding = CategoricalFeatureEmbedding(
                                                    categorical_features_metadata=self._categorical_features_metadata,
                                                    embedding_dim=self.embedding_dim,
                                                    encode=self.encode_categorical_values)
            self._num_embedded_features += self._num_categorical_features

        self.saint_encoder = SAINTEncoder(num_transformer_layers=self.num_transformer_layers,
                                          embedding_dim=self.embedding_dim,
                                          num_attention_heads=self.num_attention_heads,
                                          num_inter_sample_attention_heads=self.num_inter_sample_attention_heads,
                                          attention_dropout=self.attention_dropout,
                                          inter_sample_attention_dropout=self.inter_sample_attention_dropout,
                                          feedforward_dropout=self.feedforward_dropout,
                                          norm_epsilon=self.norm_epsilon,
                                          apply_attention_to_features=self.apply_attention_to_features,
                                          apply_attention_to_rows=self.apply_attention_to_rows,
                                          num_embedded_features=self._num_embedded_features,
                                          )
        self.flatten = layers.Flatten()
        self.norm = layers.LayerNormalization(epsilon=self.norm_epsilon)

        self.head = None
        self._is_first_batch = True
        self._is_data_in_dict_format = False

    def call(self, inputs):
        # Find the dataset's format - is it either in dictionary format or array format.
        # If inputs is an instance of dict, it's in dictionary format
        # If inputs is an instance of tuple, it's in array format
        if self._is_first_batch:
            if isinstance(inputs, dict):
                self._is_data_in_dict_format = True
            self._is_first_batch = False
        features = None
        if self.categorical_feature_embedding is not None:
            categorical_features = self.categorical_feature_embedding(inputs)
            features = categorical_features

        if self.numerical_feature_embedding is not None:
            numerical_features = self.numerical_feature_embedding(inputs)
            if features is not None:
                features = tf.concat([features, numerical_features],
                                     axis=1)
            else:
                features = numerical_features

        # Contextualize the embedded features
        features = self.saint_encoder(features)

        # Flatten the contextualized embeddings of the features
        features = self.flatten(features)

        if self._numerical_features_exists and not self.embed_numerical_features:
            # then it means that we only apply attention to categorical features
            # and only normalize the numerical features after categorical features
            # have been embedded, contextualized and flattened. Then we concatenate
            # them with the categorical features
            # Normalize numerical features
            numerical_features = tf.TensorArray(size=self._num_numerical_features,
                                                dtype=tf.float32)
            for i, (feature_name, feature_idx) in enumerate(self._numerical_features_metadata.items()):
                if self._is_data_in_dict_format:
                    feature = tf.expand_dims(inputs[feature_name], axis=1)
                else:
                    feature = tf.expand_dims(inputs[:, feature_idx], axis=1)
                feature = self.norm(feature)
                numerical_features = numerical_features.write(i, feature)
            numerical_features = tf.transpose(tf.squeeze(numerical_features.stack()))

            # Concatenate all features
            features = layers.concatenate([features, numerical_features])

        outputs = features
        if self.head is not None:
            outputs = self.head(outputs)
        return outputs


class SAINTClassifier(SAINT):
    """
    SAINTClassifier model based on the SAINT architecture proposed by
    Gowthami Somepalli et al. in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        head_units_values: `List[int] | Tuple[int]`, default [64, 32],
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
        activation_out: Activation to use in the Classification head,
            by default, `sigmoid` is used for binary and `softmax` is used
            for multi-class classification.
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
        embedding_dim: `int`, default 32,
            Embedding dimensions used in embedding numerical and categorical features.
        numerical_embedding_hidden_dim: `int` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            SAINT NumericalFeatureEmebedding layer.
        num_transformer_layer: `int`, default 6,
            Number of (SAINT) transformer layers to use in the Encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8,
            Number of attention heads to use in the MultiHeadSelfAttention layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.
        num_inter_sample_attention_heads: `int`, default 8,
            Number of heads to use in the MultiHeadInterSampleAttention that applies
            attention over rows.
        attention_dropout: `float`, default 0.1, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        inter_sample_attention_dropout: `float`, default 0.1,
            Dropout rate for MultiHeadInterSampleAttention layer that applies
            attention over rows.
        feedforward_dropout: `float`, default 0.1,
            Dropout rate to use for the dropout layer in the FeedForward block.
        norm_epsilon: `float`, default 1e-6,
            A very small number used for normalization in the `LayerNormalization` layer.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
        embed_numerical_features: `bool`, default True,
            Whether to embed the numerical features.
            If False, (SAINT) `NumericalFeatureEmbedding` layer won't be applied to
            numerical features instead they will just be normalized using `LayerNormaliztion`.
        apply_attention_to_features: `bool`, default True,
            Whether to apply attention over features using the regular `MultiHeadAttenion` layer.
        apply_attention_to_rows: `bool`, default True,
            Whether to apply attention over rows using the SAINT `MultiHeadInterSampleAttention`
            layer.
            Although it is strongly recommended to apply attention to both rows and features,
            but for experimentation's sake you can disable one of them, but NOT both at the
            same time!
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: UNITS_VALUES_TYPE = (64, 32),
                 activation_out=None,
                 features_metadata: dict = None,
                 embedding_dim: int = SAINTConfig.embedding_dim,
                 numerical_embedding_hidden_dim: int = SAINTConfig.numerical_embedding_hidden_dim,
                 num_transformer_layers: int = SAINTConfig.num_transformer_layers,
                 num_attention_heads: int = SAINTConfig.num_attention_heads,
                 num_inter_sample_attention_heads: int = SAINTConfig.num_inter_sample_attention_heads,
                 attention_dropout: float = SAINTConfig.attention_dropout,
                 inter_sample_attention_dropout: float = SAINTConfig.inter_sample_attention_dropout,
                 feedforward_dropout: float = SAINTConfig.feedforward_dropout,
                 norm_epsilon: float = SAINTConfig.norm_epsilon,
                 encode_categorical_values: bool = SAINTConfig.encode_categorical_values,
                 embed_numerical_features: bool = SAINTConfig.embed_numerical_features,
                 apply_attention_to_features: bool = SAINTConfig.apply_attention_to_features,
                 apply_attention_to_rows: bool = SAINTConfig.apply_attention_to_rows,
                 **kwargs
                 ):
        super().__init__(features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         numerical_embedding_hidden_dim=numerical_embedding_hidden_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         num_inter_sample_attention_heads=num_inter_sample_attention_heads,
                         attention_dropout=attention_dropout,
                         inter_sample_attention_dropout=inter_sample_attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         norm_epsilon=norm_epsilon,
                         encode_categorical_values=encode_categorical_values,
                         embed_numerical_features=embed_numerical_features,
                         apply_attention_to_features=apply_attention_to_features,
                         apply_attention_to_rows=apply_attention_to_rows,
                         **kwargs)
        self.num_classes = num_classes
        self.head_units_values = head_units_values
        self.activation_out = activation_out

        self.head = ClassificationHead(num_classes=self.num_classes,
                                       units_values=self.head_units_values,
                                       activation_hidden="relu",
                                       activation_out=self.activation_out,
                                       normalization="batch")


class SAINTRegressor(SAINT):
    """
    SAINTRegressor model based on the SAINT architecture proposed by
    Gowthami Somepalli et al. in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs to predict.
        head_units_values: `List[int] | Tuple[int]`, default [64, 32],
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
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
        embedding_dim: `int`, default 32,
            Embedding dimensions used in embedding numerical and categorical features.
        numerical_embedding_hidden_dim: `int` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            SAINT NumericalFeatureEmebedding layer.
        num_transformer_layer: `int`, default 6,
            Number of (SAINT) transformer layers to use in the Encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8,
            Number of attention heads to use in the MultiHeadSelfAttention layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.
        num_inter_sample_attention_heads: `int`, default 8,
            Number of heads to use in the MultiHeadInterSampleAttention that applies
            attention over rows.
        attention_dropout: `float`, default 0.1, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        inter_sample_attention_dropout: `float`, default 0.1,
            Dropout rate for MultiHeadInterSampleAttention layer that applies
            attention over rows.
        feedforward_dropout: `float`, default 0.1,
            Dropout rate to use for the dropout layer in the FeedForward block.
        norm_epsilon: `float`, default 1e-6,
            A very small number used for normalization in the `LayerNormalization` layer.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
        embed_numerical_features: `bool`, default True,
            Whether to embed the numerical features.
            If False, (SAINT) `NumericalFeatureEmbedding` layer won't be applied to
            numerical features instead they will just be normalized using `LayerNormaliztion`.
        apply_attention_to_features: `bool`, default True,
            Whether to apply attention over features using the regular `MultiHeadAttenion` layer.
        apply_attention_to_rows: `bool`, default True,
            Whether to apply attention over rows using the SAINT `MultiHeadInterSampleAttention`
            layer.
            Although it is strongly recommended to apply attention to both rows and features,
            but for experimentation's sake you can disable one of them, but NOT both at the
            same time!
    """

    def __init__(self,
                 num_outputs: int = 1,
                 head_units_values: UNITS_VALUES_TYPE = (64, 32),
                 features_metadata: dict = None,
                 embedding_dim: int = SAINTConfig.embedding_dim,
                 numerical_embedding_hidden_dim: int = SAINTConfig.numerical_embedding_hidden_dim,
                 num_transformer_layers: int = SAINTConfig.num_transformer_layers,
                 num_attention_heads: int = SAINTConfig.num_attention_heads,
                 num_inter_sample_attention_heads: int = SAINTConfig.num_inter_sample_attention_heads,
                 attention_dropout: float = SAINTConfig.attention_dropout,
                 inter_sample_attention_dropout: float = SAINTConfig.inter_sample_attention_dropout,
                 feedforward_dropout: float = SAINTConfig.feedforward_dropout,
                 norm_epsilon: float = SAINTConfig.norm_epsilon,
                 encode_categorical_values: bool = SAINTConfig.encode_categorical_values,
                 embed_numerical_features: bool = SAINTConfig.embed_numerical_features,
                 apply_attention_to_features: bool = SAINTConfig.apply_attention_to_features,
                 apply_attention_to_rows: bool = SAINTConfig.apply_attention_to_rows,
                 **kwargs
                 ):
        super().__init__(features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         numerical_embedding_hidden_dim=numerical_embedding_hidden_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         num_inter_sample_attention_heads=num_inter_sample_attention_heads,
                         attention_dropout=attention_dropout,
                         inter_sample_attention_dropout=inter_sample_attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         norm_epsilon=norm_epsilon,
                         encode_categorical_values=encode_categorical_values,
                         embed_numerical_features=embed_numerical_features,
                         apply_attention_to_features=apply_attention_to_features,
                         apply_attention_to_rows=apply_attention_to_rows,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

        self.head = RegressionHead(num_outputs=self.num_outputs,
                                   units_values=self.head_units_values,
                                   activation_hidden="relu",
                                   normalization="batch")
