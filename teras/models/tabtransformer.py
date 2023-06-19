import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers.tabtransformer import  (ColumnEmbedding,
                                          Encoder,
                                          ClassificationHead,
                                          RegressionHead
                                          )
from teras.layers.embedding import CategoricalFeatureEmbedding
from typing import List, Union, Tuple
from warnings import warn


LIST_OR_TUPLE_OF_INT = Union[List[int], Tuple[int]]
LAYER_OR_MODEL = Union[layers.Layer, keras.Model]


class TabTransformer(keras.Model):
    def __init__(self,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 6,
                 num_attention_heads: int = 8,
                 use_column_embedding: bool = True,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 norm_epsilon=1e-6,
                 categorical_features_vocabulary: dict = None,
                 encode_categorical_values: bool = True,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
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
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        self.num_categorical_features = len(categorical_features)

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

    def call(self, inputs):
        categorical_features = None
        if self.categorical_features is not None:
            categorical_features = self.categorical_feature_embedding(inputs)
            if self.use_column_embedding:
                categorical_features = self.column_embedding(categorical_features)
            # Contextualize the encoded / embedded categorical features
            categorical_features = self.encoder(categorical_features)
            # Flatten the contextualized embeddings of the categorical features
            categorical_features = self.flatten(categorical_features)

        numerical_features = []
        if self.numerical_features is not None:
            # Normalize numerical features
            for feature_name in self.numerical_features:
                numerical_features.append(self.norm(tf.expand_dims(inputs[feature_name], 1)))
            numerical_features = layers.concatenate(numerical_features, axis=1)

        # Concatenate all features
        if self.categorical_features is None:
            outputs = numerical_features
        elif self.numerical_features is None:
            outputs = categorical_features
        else:
            outputs = layers.concatenate([categorical_features, numerical_features], axis=1)

        if self.head is not None:
            outputs = self.head(outputs)
        return outputs


class TabTransformerClassifier(TabTransformer):
    def __init__(self,
                 num_classes: int = 2,
                 head_hidden_units: LIST_OR_TUPLE_OF_INT = (64, 32),
                 activation_out=None,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 6,
                 num_attention_heads: int = 8,
                 use_column_embedding: bool = True,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 norm_epsilon=1e-6,
                 categorical_features_vocabulary: dict = None,
                 encode_categorical_values: bool = True,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 **kwargs
                 ):
        super().__init__(embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         use_column_embedding=use_column_embedding,
                         norm_epsilon=norm_epsilon,
                         categorical_features_vocabulary=categorical_features_vocabulary,
                         encode_categorical_values=encode_categorical_values,
                         categorical_features=categorical_features,
                         numerical_features=numerical_features,
                         **kwargs)

        self.num_classes = num_classes
        self.head_hidden_units = head_hidden_units
        self.activation_out = activation_out
        self.head = ClassificationHead(units_hidden=self.head_hidden_units,
                                       num_classes=self.num_classes,
                                       activation_out=self.activation_out)


class TabTransformerRegressor(keras.Model):
    def __init__(self,
                 num_outputs: int = 1,
                 head_hidden_units: LIST_OR_TUPLE_OF_INT = (64, 32),
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 6,
                 num_attention_heads: int = 8,
                 use_column_embedding: bool = True,
                 attention_dropout: float = 0.,
                 feedforward_dropout: float = 0.,
                 norm_epsilon=1e-6,
                 categorical_features_vocabulary: dict = None,
                 encode_categorical_values: bool = True,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 **kwargs
                 ):
        super().__init__(embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         use_column_embedding=use_column_embedding,
                         norm_epsilon=norm_epsilon,
                         categorical_features_vocabulary=categorical_features_vocabulary,
                         encode_categorical_values=encode_categorical_values,
                         categorical_features=categorical_features,
                         numerical_features=numerical_features,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_hidden_units = head_hidden_units
        self.head = RegressionHead(units_hidden=self.head_hidden_units,
                                   units_out=self.units_out)