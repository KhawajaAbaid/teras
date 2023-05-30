import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# There are two ways of importing layers in Teras, which are demonstrated below.
# Layer names in teras.layers module are usually prepended
# with the architecture name in which they were proposed, in this case TabTransformer.
# This is done to,
#   1. Identify which architecture the layer was proposed in.
#   2. Avoid name conflicts.
from teras.layers.tabtransformer import CategoricalFeatureEmbedding, ColumnEmbedding
from teras.layers import TabTransformerEncoder as Encoder
from teras.layers import TabTransformerClassificationHead as ClassificationHead
from teras.layers.tabtransformer import RegressionHead
from typing import List


class TabTransformerClassifier(keras.Model):
    def __init__(self,
                 num_classes=2,
                 embedding_dim=32,
                 num_transformer_layers=6,
                 num_attention_heads=8,
                 categorical_features_vocab=None,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 use_column_embedding: bool = True,
                 attention_dropout=0.,
                 feedforward_dropout=0.,
                 norm_epsilon=1e-6,
                 head_hidden_units: List[int] = [64, 32],
                 activation_out=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        assert categorical_features_vocab is not None, \
            ("You need to pass categorical_features_vocab to TabTransformer."
             " Use teras.utils.get_categorical_features_vocab(dataset, list_of_categorical_feature_names)")
        self.categorical_features_vocab = categorical_features_vocab
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.use_column_embedding = use_column_embedding
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.head_hidden_units = head_hidden_units
        self.activation_out = activation_out

        if self.activation_out is None:
            self.activation_out = "sigmoid" if self.num_classes == 1 else "softmax"

        self.num_categorical_features = len(categorical_features)
        # self.input_dim = self.num_categorical_features + len(self.numerical_features)
        self.input_dim = self.num_categorical_features
        self.categorical_feature_embedding = CategoricalFeatureEmbedding(
                                                        categorical_features=self.categorical_features,
                                                        categorical_features_vocab=self.categorical_features_vocab,
                                                        embedding_dim=self.embedding_dim
                                                        )
        if self.use_column_embedding:
            self.column_embedding = ColumnEmbedding(embedding_dim=self.embedding_dim,
                                                    num_categorical_features=self.num_categorical_features)
        self.encoder = Encoder(num_transformer_layers=self.num_transformer_layers,
                               num_heads=self.num_attention_heads,
                               embedding_dim=self.embedding_dim,
                               attention_dropout=self.attention_dropout,
                               feedforward_dropout=self.feedforward_dropout)
        self.flatten = keras.layers.Flatten()
        self.norm = keras.layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.head = ClassificationHead(units_hidden=self.head_hidden_units,
                                                      num_classes=self.num_classes,
                                                      activation_out=self.activation_out)

    def call(self, inputs, training=None, mask=None):
        categorical_features_embeddings = self.categorical_feature_embedding(inputs)
        if self.use_column_embedding:
             categorical_features_embeddings = self.column_embedding(categorical_features_embeddings)
        x = categorical_features_embeddings
        # Contextualize the encoded / embedded categorical features
        x = self.encoder(x)
        # Flatten the contextualized embeddings of the categorical features
        categorical_features = self.flatten(x)
        normalized_numerical_features = []
        if self.numerical_features:
            # Normalize numerical features
            for num_feat in self.numerical_features:
                normalized_numerical_features.append(self.norm(tf.expand_dims(inputs[num_feat], 1)))
            normalized_numerical_features = layers.concatenate(normalized_numerical_features, axis=1)

        # Concatenate all features
        features = categorical_features
        if self.numerical_features:
            features = layers.concatenate([categorical_features, normalized_numerical_features], axis=1)
        out = self.head(features)
        return out


class TabTransformerRegressor(keras.Model):
    def __init__(self,
                 units_out=1,
                 embedding_dim=32,
                 num_transformer_layers=6,
                 num_attention_heads=8,
                 categorical_features_vocab=None,
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 use_column_embedding: bool = True,
                 attention_dropout=0.,
                 feedforward_dropout=0.,
                 norm_epsilon=1e-6,
                 head_hidden_units: List[int] = [64, 32],
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.units_out = units_out
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        assert categorical_features_vocab is not None, \
            ("You need to pass categorical_features_vocab to TabTransformer."
             " Use teras.utils.get_categorical_features_vocab(dataset, list_of_categorical_feature_names)")
        self.categorical_features_vocab = categorical_features_vocab
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.use_column_embedding = use_column_embedding
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.head_hidden_units = head_hidden_units

        self.num_categorical_features = len(categorical_features)
        self.input_dim = self.num_categorical_features + len(self.numerical_features)

        self.categorical_features_embeddings = CategoricalFeatureEmbedding(
                                                        categorical_features=self.categorical_features,
                                                        categorical_features_vocab=self.categorical_features_vocab
                                                        )
        if self.use_column_embedding:
            self.column_embedding = ColumnEmbedding(embedding_dim=self.embedding_dim,
                                                    num_categorical_features=self.num_categorical_features)
        self.encoder = Encoder(num_transformer_layers=self.num_transformer_layers,
                               num_heads=self.num_attention_heads,
                               embedding_dim=self.embedding_dim,
                               attention_dropout=self.attention_dropout,
                               feedforward_dropout=self.feedforward_dropout)
        self.flatten = keras.layers.Flatten()
        self.norm = keras.layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.head = RegressionHead(units_hidden=self.head_hidden_units,
                                              units_out=self.units_out)

    def call(self, inputs, training=None, mask=None):
        categorical_features_embeddings = self.categorical_features_embeddings(inputs)
        if self.use_column_embedding:
             categorical_features_embeddings += self.column_embedding(categorical_features_embeddings)
        x = categorical_features_embeddings

        # Contextualize the encoded / embedded categorical features
        x = self.encoder(x)

        # Flatten the contextualized embeddings of the categorical features
        categorical_features = self.flatten(x)

        normalized_numerical_features = []
        if self.numerical_features:
            # Normalize numerical features
            for num_feat in self.numerical_features:
                normalized_numerical_features.append(self.norm(tf.expand_dims(inputs[num_feat], 1)))
            normalized_numerical_features = layers.concatenate(normalized_numerical_features, axis=1)

        # Concatenate all features
        features = categorical_features
        if self.numerical_features:
            features = layers.concatenate([categorical_features, normalized_numerical_features], axis=1)

        out = self.head(features)
        return out