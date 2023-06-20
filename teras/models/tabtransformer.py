import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers.tabtransformer import (ColumnEmbedding,
                                         Encoder,
                                         ClassificationHead,
                                         RegressionHead
                                         )
from teras.layers.embedding import CategoricalFeatureEmbedding
from typing import List, Union, Tuple
from warnings import warn
from teras.config.tabtransformer import TabTransformerConfig


LIST_OR_TUPLE_OF_INT = Union[List[int], Tuple[int]]
LAYER_OR_MODEL = Union[layers.Layer, keras.Model]


class TabTransformer(keras.Model):
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
        embedding_dim: `int`, default 32, Dimensionality of the learnable
            feature embeddings for categorical features.
        num_transformer_layers: `int`, default 6, Number of transformer layers
            to use in the encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8, Number of attention heads to use
            in the MultiHeadSelfAttention layer that is part of the transformer
            layer which in turn is part of the encoder.
        attention_dropout: `float`, default 0.0, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        feedforward_dropout: `float`, default 0.0, Dropout rate to use for the
            dropout layer in the FeedForward block.
        norm_epsilon: `float`, default 1e-6, A very small number used for normalization
            in the LayerNormalization layer.
        use_column_embedding: `bool`, default True, Whether to use the novel ColumnEmbedding
            layer proposed in the TabTransformer architecture for the categorical features.
            The ColumnEmbedding layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.
        categorical_features_vocabulary: `dict`, Vocabulary of categorical feature.
            Vocabulary is simply a dictionary where feature name maps
            to a tuple of feature index and a list of unique values in the feature.
            You can get this vocabulary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features)`
            If None, dataset will be assumed to contain no categorical features and
            hence CategoricalFeatureEmbedding layer won't be applied.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
"""
    def __init__(self,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 categorical_features_vocabulary: dict = TabTransformerConfig.categorical_features_vocabulary,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        if categorical_features_vocabulary is None:
            warn("""
            No value for `categorical_features_vocabulary` was passed. 
            It is assumed that the dataset doesn't contain any categorical features,
            hence CategoricalFeaturesEmbedding won't be applied. "
            If your dataset does contain categorical features and you must pass the
            `categorical_features_vocabulary` for better performance and to avoid unexpected results.
            You can get this vocabulary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features)`
                 """)

        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.use_column_embedding = use_column_embedding
        self.norm_epsilon = norm_epsilon
        self.categorical_features_vocabulary = categorical_features_vocabulary
        self.encode_categorical_values = encode_categorical_values

        self.num_categorical_features = len(self.categorical_features_vocabulary)

        self.categorical_feature_embedding = CategoricalFeatureEmbedding(
            categorical_features_vocabulary=self.categorical_features_vocabulary,
            embedding_dim=self.embedding_dim
        )

        self.column_embedding = ColumnEmbedding(embedding_dim=self.embedding_dim,
                                                num_categorical_features=self.num_categorical_features)

        self.encoder = Encoder(num_transformer_layers=self.num_transformer_layers,
                               num_heads=self.num_attention_heads,
                               embedding_dim=self.embedding_dim,
                               attention_dropout=self.attention_dropout,
                               feedforward_dropout=self.feedforward_dropout)
        self.flatten = layers.Flatten()
        self.norm = layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.head = None

        self._numerical_features_exist = True
        self._categorical_features_exist = True

        # Since we already have categorical features names and indices in the
        # categorical features vocabulary -- but we also do need this info for
        # numerical features -- names in the case of dictionary format inputs
        # and idx in the case of regular array format inputs.
        # Fortunately we can find out both in the call method.
        # If inputs are dict, we can simply extract the keys and delete out the
        # categorical features names to get numerical features names
        # And if inputs are arrays, we can construct a input_dim array, in the
        # range 0 - input_dim and delete out numbers that correspond to catgorical
        # feature indices to get the indices for the numerical features.
        self._categorical_features_names = None
        self._categorical_features_idx = None
        if self.categorical_features_vocabulary is not None:
            self._categorical_features_idx = set(map(lambda x: x[0], categorical_features_vocabulary.values()))
            self._categorical_features_names = set(categorical_features_vocabulary.keys())
        else:
            self._categorical_features_exist = False
        self._numerical_features_names = None
        self._numerical_features_idx = None

        self._is_first_batch = True
        self._is_data_in_dict_format = False

    def call(self, inputs):
        if self._is_first_batch:
            if isinstance(inputs, dict):
                self._is_data_in_dict_format = True
                all_features_names = set(inputs.keys())
                if self._categorical_features_names is None:
                    # Then there are only numerical features in the dataset
                    self._numerical_features_names = all_features_names
                else:
                    # Otherwise there are definitely categorical features but may or may not be
                    # numerical features -- why? let's see.
                    self._numerical_features_names = all_features_names - self._categorical_features_names
                    if len(self._numerical_features_names) == 0:
                        # There are no numerical features
                        self._numerical_features_exist = False
            else:
                # otherwise the inputs must be in regular arrays format
                all_features_idx = set(range(tf.shape(inputs)[1]))
                if self._categorical_features_idx is None:
                    self._numerical_features_idx = all_features_idx
                else:
                    self._numerical_features_idx = all_features_idx - self._categorical_features_idx
                if len(self._numerical_features_idx) == 0:
                    self._numerical_features_exist = False
            self._is_first_batch = False

        categorical_features = None
        if self._categorical_features_exist:
            # The categorical feature embedding layer takes care of handling
            # different input data types and features names nad indices
            categorical_features = self.categorical_feature_embedding(inputs)
            if self.use_column_embedding:
                categorical_features = self.column_embedding(categorical_features)
            # Contextualize the embedded categorical features
            categorical_features = self.encoder(categorical_features)
            # Flatten the contextualized embeddings of the categorical features
            categorical_features = self.flatten(categorical_features)

        numerical_features = []
        if self._numerical_features_exist:
            # Normalize numerical features
            if self._is_data_in_dict_format:
                for feature_name in self._numerical_features_names:
                    numerical_features.append(self.norm(tf.expand_dims(inputs[feature_name], 1)))
            else:
                for feature_idx in self._numerical_features_idx:
                    numerical_features.append(self.norm(tf.expand_dims(inputs[:, feature_idx], 1)))
            numerical_features = layers.concatenate(numerical_features, axis=1)

        # Concatenate all features
        if not self._categorical_features_exist:
            outputs = numerical_features
        elif not self._numerical_features_exist:
            outputs = categorical_features
        else:
            outputs = layers.concatenate([categorical_features, numerical_features], axis=1)

        if self.head is not None:
            outputs = self.head(outputs)

        return outputs


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
        num_classes: `int`, default 1, Number of classes to predict.
        head_hidden_units: `List[int]`, default [64, 32], Hidden units to
            use in the Classification head. For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
        activation_out: Activation to use in the Classification head,
            by default, `sigmoid` is used for binary and `softmax` is used
            for multi-class classification.
        embedding_dim: `int`, default 32, Dimensionality of the learnable
            feature embeddings for categorical features.
        num_transformer_layers: `int`, default 6, Number of transformer layers
            to use in the encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8, Number of attention heads to use
            in the MultiHeadSelfAttention layer that is part of the transformer
            layer which in turn is part of the encoder.
        attention_dropout: `float`, default 0.0, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        feedforward_dropout: `float`, default 0.0, Dropout rate to use for the
            dropout layer in the FeedForward block.
        norm_epsilon: `float`, default 1e-6, A very small number used for normalization
            in the LayerNormalization layer.
        use_column_embedding: `bool`, default True, Whether to use the novel ColumnEmbedding
            layer proposed in the TabTransformer architecture for the categorical features.
            The ColumnEmbedding layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.
        categorical_features_vocabulary: `dict`, Vocabulary of categorical feature.
            Vocabulary is simply a dictionary where feature name maps
            to a tuple of feature index and a list of unique values in the feature.
            You can get this vocabulary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features)`
            If None, dataset will be assumed to contain no categorical features and
            hence CategoricalFeatureEmbedding layer won't be applied.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_hidden_units: LIST_OR_TUPLE_OF_INT = (64, 32),
                 activation_out=None,
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 categorical_features_vocabulary: dict = TabTransformerConfig.categorical_features_vocabulary,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 **kwargs
                 ):
        super().__init__(embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         norm_epsilon=norm_epsilon,
                         use_column_embedding=use_column_embedding,
                         categorical_features_vocabulary=categorical_features_vocabulary,
                         encode_categorical_values=encode_categorical_values,
                         **kwargs)

        self.num_classes = num_classes
        self.head_hidden_units = head_hidden_units
        self.activation_out = activation_out
        self.head = ClassificationHead(num_classes=self.num_classes,
                                       units_hidden=self.head_hidden_units,
                                       activation_out=self.activation_out)


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
        num_outputs: `int`, default 1, Number of regression outputs to predict.
        head_hidden_units: `List[int]`, default [64, 32], Hidden units to
            use in the Regression head. For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
        embedding_dim: `int`, default 32, Dimensionality of the learnable
            feature embeddings for categorical features.
        num_transformer_layers: `int`, default 6, Number of transformer layers
            to use in the encoder.
            The encoder is used to contextualize the learned feature embeddings.
        num_attention_heads: `int`, default 8, Number of attention heads to use
            in the MultiHeadSelfAttention layer that is part of the transformer
            layer which in turn is part of the encoder.
        attention_dropout: `float`, default 0.0, Dropout rate to use in the
            MultiHeadSelfAttention layer in the transformer layer.
        feedforward_dropout: `float`, default 0.0, Dropout rate to use for the
            dropout layer in the FeedForward block.
        norm_epsilon: `float`, default 1e-6, A very small number used for normalization
            in the LayerNormalization layer.
        use_column_embedding: `bool`, default True, Whether to use the novel ColumnEmbedding
            layer proposed in the TabTransformer architecture for the categorical features.
            The ColumnEmbedding layer is an alternative to positional encoding that is applied
            in the Transformers in Natural Langauge Processing application settings.
        categorical_features_vocabulary: `dict`, Vocabulary of categorical feature.
            Vocabulary is simply a dictionary where feature name maps
            to a tuple of feature index and a list of unique values in the feature.
            You can get this vocabulary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features)`
            If None, dataset will be assumed to contain no categorical features and
            hence CategoricalFeatureEmbedding layer won't be applied.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 head_hidden_units: LIST_OR_TUPLE_OF_INT = (64, 32),
                 embedding_dim: int = TabTransformerConfig.embedding_dim,
                 num_transformer_layers: int = TabTransformerConfig.num_transformer_layers,
                 num_attention_heads: int = TabTransformerConfig.num_attention_heads,
                 attention_dropout: float = TabTransformerConfig.attention_dropout,
                 feedforward_dropout: float = TabTransformerConfig.feedforward_dropout,
                 norm_epsilon: float = TabTransformerConfig.norm_epsilon,
                 use_column_embedding: bool = TabTransformerConfig.use_column_embedding,
                 categorical_features_vocabulary: dict = TabTransformerConfig.categorical_features_vocabulary,
                 encode_categorical_values: bool = TabTransformerConfig.encode_categorical_values,
                 **kwargs
                 ):
        super().__init__(embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         norm_epsilon=norm_epsilon,
                         use_column_embedding=use_column_embedding,
                         categorical_features_vocabulary=categorical_features_vocabulary,
                         encode_categorical_values=encode_categorical_values,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_hidden_units = head_hidden_units
        self.head = RegressionHead(num_outputs=self.num_outputs,
                                   units_hidden=self.head_hidden_units)
