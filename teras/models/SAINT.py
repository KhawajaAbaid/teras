import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from warnings import warn
from typing import List
from teras.layers import (SAINTEncoder,
                          SAINTNumericalFeaturesEmbedding,
                          SAINTCategoricalFeaturesEmbedding,
                          SAINTClassificationHead,
                          SAINTRegressionHead
                          )


class SAINTClassifier(keras.Model):
    """
    SAINTClassifier model based on the architecture proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_classes: Number of classes to predict
        categorical_features: List of names of categorical features in the dataset
        numerical_features: List of names of numerical/continuous features in the dataset
        embedding_dim: Embeddings Dimensionality for both Numerical and Catergorical features.
        num_transformer_layer: Number of transformer layers to use in the encoder
        num_heads_feature_attn: Number of heads to use in the
            MultiHeadAttention that will be applied over features
        num_heads_inter_sample_attn: Number of heads to use in the
            MultiHeadInterSampleAttention that will be applied over rows
        embedding_dim: Embedding dimensions in the MultiHeadAttention layer
        feature_attention_dropout: Dropout rate for MultiHeadAttention over features
        inter_sample_attention_dropout: Dropout rate for MultiInterSample HeadAttention over rows        feedforward_dropout: Dropout rate to use in the FeedForward layer
        feedforward_dropout: Dropout rate for FeedForward layer
        norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
        numerical_embedding_type: Type of embedding to apply to numerical features.
            Currently only support "MLP", but you can pass None to use numerical features as is.
            Note in case of None, numerical features will be normalized.
        numerical_embedding_hidden_dim: Hidden dimensions for MLP based Numerical Embedding.
        use_inter_sample_attention: Whether to use inter sample attention
        rows_only: When use_inter_sample_attention is True, this parameter determines whether to
            apply attention over just rows (when True) or over both rows and columns (when False).
            Defaults to False.
        num_features: Number of features in the input
        use_inter_sample_attention: Whether to use inter_sample attention
                (without inter sample attention, it's pretty much TabTransformer)
        apply_attention_to_rows_only: Will only be used if use_inter_sample_attention is True. In that case, attention
                won't be applied to features.
        units_hidden: List of units to use in hidden dense layers.
            Number of hidden dense layers will be equal to the length of units_hidden list.
        activation_out: Activation to apply over outputs.
            By default, sigmoid is used for binary while softmax for multiclass classification.
    """
    def __init__(self,
                 num_classes=None,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 embedding_dim=32,
                 num_transformer_layers=6,
                 num_heads_feature_attention=8,
                 num_heads_inter_sample_attention=8,
                 categorical_features_vocab=None,
                 feature_attention_dropout=0.1,
                 inter_sample_attention_dropout=0.1,
                 feedforward_dropout=0.1,
                 norm_epsilon=1e-6,
                 numerical_embedding_type="MLP",
                 numerical_embedding_hidden_dim=16,
                 use_inter_sample_attention=True,
                 apply_attention_to_rows_only=False,
                 head_hidden_units: List[int] = [64, 32],
                 activation_out=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads_feature_attention = num_heads_feature_attention
        self.num_heads_inter_sample_attention = num_heads_inter_sample_attention
        assert categorical_features_vocab is not None, ("You need to pass categorical_features_vocab to SAINTTransformer"
                            "Use SAINTTransformer.utility.get_categorical_features_vocab(inputs, categorical_features)")
        self.categorical_features_vocab = categorical_features_vocab
        self.feature_attention_dropout = feature_attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.numerical_embedding_type = numerical_embedding_type
        self.numerical_embedding_hidden_dim = numerical_embedding_hidden_dim
        self.use_inter_sample_attention = use_inter_sample_attention
        self.apply_attention_to_rows_only = apply_attention_to_rows_only
        self.head_hidden_units = head_hidden_units
        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = 'sigmoid' if self.num_classes == 1 else 'softmax'


        self.num_categorical_features = len(categorical_features)
        self.num_numerical_features = len(numerical_features)

        # Numerical/Continuous Features Embedding Layers
        if self.numerical_embedding_type == "MLP":
            self.numerical_embedding = SAINTNumericalFeaturesEmbedding(numerical_features=self.numerical_features,
                                                                       hidden_dim=self.numerical_embedding_hidden_dim,
                                                                       embedding_dim=self.embedding_dim)
            self.num_features = self.num_numerical_features + self.num_categorical_features
        else:
            warn("numerical_embedding_type isn't set to 'MLP', hence numerical features won't be passed through attention.")
            self.num_features = self.num_categorical_features


        # Categorical Features Embedding Layers
        self.input_dim = self.num_categorical_features + self.num_numerical_features
        self.lookup_tables, self.embedding_layers = self.get_lookup_tables_and_embedding_layers()
        self.categorical_features_embedding = SAINTCategoricalFeaturesEmbedding(categorical_features=self.categorical_features,
                                                                                lookup_tables=self.lookup_tables,
                                                                                embedding_layers=self.embedding_layers)

        self.saint_encoder = SAINTEncoder(num_transformer_layers=self.num_transformer_layers,
                                          num_heads_feature_attn=self.num_heads_feature_attention,
                                          num_heads_inter_sample_attn=self.num_heads_inter_sample_attention,
                                          embedding_dim=self.embedding_dim,
                                          feature_attention_dropout=self.feature_attention_dropout,
                                          inter_sample_attention_dropout=self.inter_sample_attention_dropout,
                                          feedforward_dropout=self.feedforward_dropout,
                                          norm_epsilon=self.norm_epsilon,
                                          use_inter_sample_attention=self.use_inter_sample_attention,
                                          apply_attention_to_rows_only=self.apply_attention_to_rows_only,
                                          num_features=self.num_features)
        self.flatten = layers.Flatten()
        self.norm = layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.head = SAINTClassificationHead(num_classes=self.num_classes,
                                            units_hidden=[64, 32],
                                            activation_out=self.activation_out)

    def get_lookup_tables_and_embedding_layers(self):
        """Lookup tables and embedding layers for each categorical feature"""
        lookup_tables = {}
        embedding_layers = {}
        for feature in self.categorical_features:
            vocab = self.categorical_features_vocab[feature]
            # Lookup Table to convert string values to integer indices
            lookup = layers.StringLookup(vocabulary=vocab,
                                                mask_token=None,
                                                num_oov_indices=0,
                                                output_mode="int"
                                            )
            lookup_tables[feature] = lookup

            # Create embedding layer
            embedding = layers.Embedding(input_dim=len(vocab),
                                               output_dim=self.embedding_dim)
            embedding_layers[feature] = embedding
        return lookup_tables, embedding_layers

    def call(self, inputs):
        categorical_features_embeddings = self.categorical_features_embedding(inputs)
        feature_embeddings = categorical_features_embeddings
        feature_embeddings = tf.squeeze(feature_embeddings, axis=2)
        feature_embeddings = tf.transpose(feature_embeddings, perm=[1, 0, 2])
        if self.numerical_embedding_type is not None:
            numerical_features_embeddings = self.numerical_embedding(inputs)
            numerical_features_embeddings = tf.transpose(numerical_features_embeddings, perm=[1, 0, 2])
            feature_embeddings = tf.concat([feature_embeddings, numerical_features_embeddings],
                                            axis=1)

        # Contextualize the encoded / embedded categorical features
        contextualized_embeddings = self.saint_encoder(feature_embeddings)

        # Flatten the contextualized embeddings of the features
        features = self.flatten(contextualized_embeddings)

        if self.numerical_embedding_type is None:
            # if it's none, then it means that we only apply attention to categorical features
            # and the numerical features are as is. So we normalize them and concatenate with
            # the categorical features
            # Normalize numerical features
            numerical_features = self.norm(inputs[self.numerical_features])
            # Concatenate all features
            features = layers.concatenate([features, numerical_features])
        out = self.head(features)
        return out


class SAINTRegressor(keras.Model):
    """
    SAINTRegressor model based on the architecture proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        units_out: Number of regression outputs
        categorical_features: List of names of categorical features in the dataset
        numerical_features: List of names of numerical/continuous features in the dataset
        embedding_dim: Embeddings Dimensionality for both Numerical and Catergorical features.
        num_transformer_layer: Number of transformer layers to use in the encoder
        num_heads_feature_attn: Number of heads to use in the
            MultiHeadAttention that will be applied over features
        num_heads_inter_sample_attn: Number of heads to use in the
            MultiHeadInterSampleAttention that will be applied over rows
        embedding_dim: Embedding dimensions in the MultiHeadAttention layer
        feature_attention_dropout: Dropout rate for MultiHeadAttention over features
        inter_sample_attention_dropout: Dropout rate for MultiInterSample HeadAttention over rows        feedforward_dropout: Dropout rate to use in the FeedForward layer
        feedforward_dropout: Dropout rate for FeedForward layer
        norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
        numerical_embedding_type: Type of embedding to apply to numerical features.
            Currently only support "MLP", but you can pass None to use numerical features as is.
            Note in case of None, numerical features will be normalized.
        numerical_embedding_hidden_dim: Hidden dimensions for MLP based Numerical Embedding.
        use_inter_sample_attention: Whether to use inter sample attention
        rows_only: When use_inter_sample_attention is True, this parameter determines whether to
            apply attention over just rows (when True) or over both rows and columns (when False).
            Defaults to False.
        num_features: Number of features in the input
        use_inter_sample_attention: Whether to use inter_sample attention
                (without inter sample attention, it's pretty much TabTransformer)
        apply_attention_to_rows_only: Will only be used if use_inter_sample_attention is True. In that case, attention
                won't be applied to features.
        units_hidden: List of units to use in hidden dense layers.
            Number of hidden dense layers will be equal to the length of units_hidden list.
    """

    def __init__(self,
                 units_out=1,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 embedding_dim=32,
                 num_transformer_layers=6,
                 num_heads_feature_attention=8,
                 num_heads_inter_sample_attention=8,
                 categorical_features_vocab=None,
                 feature_attention_dropout=0.1,
                 inter_sample_attention_dropout=0.1,
                 feedforward_dropout=0.1,
                 norm_epsilon=1e-6,
                 numerical_embedding_type="MLP",
                 numerical_embedding_hidden_dim=16,
                 use_inter_sample_attention=True,
                 apply_attention_to_rows_only=False,
                 head_hidden_units: List[int] = [64, 32],
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.units_out = units_out
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads_feature_attention = num_heads_feature_attention
        self.num_heads_inter_sample_attention = num_heads_inter_sample_attention
        assert categorical_features_vocab is not None, (
            "You need to pass categorical_features_vocab to SAINTTransformer"
            "Use SAINTTransformer.utility.get_categorical_features_vocab(inputs, categorical_features)")
        self.categorical_features_vocab = categorical_features_vocab
        self.feature_attention_dropout = feature_attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.numerical_embedding_type = numerical_embedding_type
        self.numerical_embedding_hidden_dim = numerical_embedding_hidden_dim
        self.use_inter_sample_attention = use_inter_sample_attention
        self.apply_attention_to_rows_only = apply_attention_to_rows_only
        self.head_hidden_units = head_hidden_units
        if self.activation_out is None:
            self.activation_out = 'sigmoid' if self.num_classes == 1 else 'softmax'

        self.num_categorical_features = len(categorical_features)
        self.num_numerical_features = len(numerical_features)

        # Numerical/Continuous Features Embedding Layers
        if self.numerical_embedding_type == "MLP":
            self.numerical_embedding = SAINTNumericalFeaturesEmbedding(numerical_features=self.numerical_features,
                                                                       hidden_dim=self.numerical_embedding_hidden_dim,
                                                                       embedding_dim=self.embedding_dim)
            self.num_features = self.num_numerical_features + self.num_categorical_features
        else:
            warn(
                "numerical_embedding_type isn't set to 'MLP', hence numerical features won't be passed through attention.")
            self.num_features = self.num_categorical_features

        # Categorical Features Embedding Layers
        self.input_dim = self.num_categorical_features + self.num_numerical_features
        self.lookup_tables, self.embedding_layers = self.get_lookup_tables_and_embedding_layers()
        self.categorical_features_embedding = SAINTCategoricalFeaturesEmbedding(
            categorical_features=self.categorical_features,
            lookup_tables=self.lookup_tables,
            embedding_layers=self.embedding_layers)

        self.saint_encoder = SAINTEncoder(num_transformer_layers=self.num_transformer_layers,
                                          num_heads_feature_attn=self.num_heads_feature_attention,
                                          num_heads_inter_sample_attn=self.num_heads_inter_sample_attention,
                                          embedding_dim=self.embedding_dim,
                                          feature_attention_dropout=self.feature_attention_dropout,
                                          inter_sample_attention_dropout=self.inter_sample_attention_dropout,
                                          feedforward_dropout=self.feedforward_dropout,
                                          norm_epsilon=self.norm_epsilon,
                                          use_inter_sample_attention=self.use_inter_sample_attention,
                                          apply_attention_to_rows_only=self.apply_attention_to_rows_only,
                                          num_features=self.num_features)
        self.flatten = layers.Flatten()
        self.norm = layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.head = SAINTRegressionHead(units_out=self.units,
                                        units_hidden=[64, 32])

    def get_lookup_tables_and_embedding_layers(self):
        """Lookup tables and embedding layers for each categorical feature"""
        lookup_tables = {}
        embedding_layers = {}
        for feature in self.categorical_features:
            vocab = self.categorical_features_vocab[feature]
            # Lookup Table to convert string values to integer indices
            lookup = layers.StringLookup(vocabulary=vocab,
                                         mask_token=None,
                                         num_oov_indices=0,
                                         output_mode="int"
                                         )
            lookup_tables[feature] = lookup

            # Create embedding layer
            embedding = layers.Embedding(input_dim=len(vocab),
                                         output_dim=self.embedding_dim)
            embedding_layers[feature] = embedding
        return lookup_tables, embedding_layers

    def call(self, inputs):
        categorical_features_embeddings = self.categorical_features_embedding(inputs)
        feature_embeddings = categorical_features_embeddings
        feature_embeddings = tf.squeeze(feature_embeddings, axis=2)
        feature_embeddings = tf.transpose(feature_embeddings, perm=[1, 0, 2])
        if self.numerical_embedding_type is not None:
            numerical_features_embeddings = self.numerical_embedding(inputs)
            numerical_features_embeddings = tf.transpose(numerical_features_embeddings, perm=[1, 0, 2])
            feature_embeddings = tf.concat([feature_embeddings, numerical_features_embeddings],
                                           axis=1)

        # Contextualize the encoded / embedded categorical features
        contextualized_embeddings = self.saint_encoder(feature_embeddings)

        # Flatten the contextualized embeddings of the features
        features = self.flatten(contextualized_embeddings)

        if self.numerical_embedding_type is None:
            # if it's none, then it means that we only apply attention to categorical features
            # and the numerical features are as is. So we normalize them and concatenate with
            # the categorical features
            # Normalize numerical features
            numerical_features = self.norm(inputs[self.numerical_features])
            # Concatenate all features
            features = layers.concatenate([features, numerical_features])
        out = self.head(features)
        return out