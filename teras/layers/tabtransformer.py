import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from teras.layers import GEGLU
# import numpy as np


# class CategoricalFeatureEmbedding(layers.Layer):
#     """
#     CategoricalFeatureEmbedding layer that encodes categorical features into
#     categorical feature embeddings.
#
#     Args:
#         categorical_features: List of categorical features/columns names in the dataset
#         categorical_features_vocab: Vocabulary of values of each categorical feature.
#             You can get this vocabulary by calling
#             teras.utils.get_categorical_features_vocab(dataset, list_of_categorical_feature_names)
#     """
#     def __init__(self,
#                  categorical_features,
#                  categorical_features_vocab,
#                  embedding_dim=32,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.categorical_features = categorical_features
#         self.categorical_features_vocab = categorical_features_vocab
#         self.embedding_dim = embedding_dim
#         self.lookup_tables, self.embedding_layers = self._get_lookup_tables_and_embedding_layers()
#         self.concat = keras.layers.Concatenate(axis=1)
#
#     def _get_lookup_tables_and_embedding_layers(self):
#         """Lookup tables and embedding layers for each categorical feature"""
#         lookup_tables = {}
#         embedding_layers = {}
#         for feature in self.categorical_features:
#             vocab = self.categorical_features_vocab[feature]
#             # Lookup Table to convert string values to integer indices
#             lookup = keras.layers.StringLookup(vocabulary=vocab,
#                                                mask_token=None,
#                                                num_oov_indices=0,
#                                                output_mode="int"
#                                                )
#             lookup_tables[feature] = lookup
#
#             # Create embedding layer
#             embedding = keras.layers.Embedding(input_dim=len(vocab),
#                                                output_dim=self.embedding_dim)
#             embedding_layers[feature] = embedding
#
#         return lookup_tables, embedding_layers
#
#     def call(self, inputs):
#         # Encode and embedd categorical features
#         categorical_features_embeddings = []
#         for feature_id, feature in enumerate(self.categorical_features):
#             lookup = self.lookup_tables[feature]
#             embedding = self.embedding_layers[feature]
#             # Convert string input values to integer indices
#             feature = tf.expand_dims(inputs[feature], 1)
#             encoded_feature = lookup(feature)
#             # Convert index values to embedding representations
#             encoded_feature = embedding(encoded_feature)
#             categorical_features_embeddings.append(encoded_feature)
#
#         categorical_features_embeddings = self.concat(categorical_features_embeddings)
#         return categorical_features_embeddings


class ColumnEmbedding(layers.Layer):
    """
    ColumnEmbedding layer as proposed by Xin Huang et al. in the paper
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

    Reference(s):
        https://arxiv.org/abs/2012.06678

    Args:
        TODO
    """
    def __init__(self,
                 embedding_dim=32,
                 num_categorical_features=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_categorical_features = num_categorical_features
        self.column_embedding = keras.layers.Embedding(input_dim=self.num_categorical_features,
                                                       output_dim=self.embedding_dim)
        self.column_indices = tf.range(start=0,
                                       limit=self.num_categorical_features,
                                       delta=1)
        self.column_indices = tf.cast(self.column_indices, dtype="float32")

    def call(self, inputs):
        """
        Args:
            inputs: Embeddings of categorical features encoded by CategoricalFeatureEmbedding layer
        """
        return inputs + self.column_embedding(self.column_indices)


# class FeedForward(layers.Layer):
#     """
#     FeedForward layer as proposed by Xin Huang et al. in the paper
#     TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
#     The output of MutliHeadAttention layer passed through two dense layers.
#     The first layer expands the `embedding dimensions` to four times its size
#     and the second layer projects it back to its original size.
#
#     Reference(s):
#         https://arxiv.org/abs/2012.06678
#
#     Args:
#         embedding_dim: Embedding dimensions
#         multiplier: Multiplier to expand the output dimensions of MultiHeadAttention using a dense layer.
#             The authors propose a multiplier of 4, which is default.
#         dropout: Dropout rate.
#         activation: Activation function to use in the first dense layer.
#             By default, GEGLU activation is used, which isn't offered by Keras by is offered by Teras.
#             You can import it using teras.layers.GEGLU for your own use.
#     """
#     def __init__(self,
#                  embedding_dim=16,
#                  multiplier=4,
#                  dropout=0.,
#                  activation="geglu",
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.embedding_dim = embedding_dim
#         self.multiplier = multiplier
#         self.dropout = dropout
#         self.activation = GEGLU() if activation.lower() == "geglu" else activation
#         self.dense_1 = keras.layers.Dense(self.embedding_dim * self.multiplier,
#                                           activation=self.activation)
#         self.dropout = keras.layers.Dropout(self.dropout)
#         self.dense_2 = keras.layers.Dense(self.embedding_dim)
#
#     def call(self, inputs):
#         x = self.dense_1(inputs)
#         x = self.dropout(x)
#         x = self.dense_2(x)
#         return x


# class Transformer(layers.Layer):
#     """
#     Transformer layer as proposed by Xin Huang et al. in the paper
#     TabTransformer: Tabular Data Modeling Using Contextual Embeddings
#
#     Reference(s):
#         https://arxiv.org/abs/2012.06678
#
#     Args:
#         num_heads: Number of heads to use in the MultiHeadAttention layer
#         embedding_dim: Embedding dimensions in the MultiHeadAttention layer
#         attention_dropout: Dropout rate to use in the MultiHeadAttention layer
#         feedforward_dropout: Dropout rate to use in the FeedForward layer
#         norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
#     """
#     def __init__(self,
#                  num_heads,
#                  embedding_dim,
#                  attention_dropout,
#                  feedforward_dropout,
#                  norm_epsilon=1e-6,
#                  **kwagrs):
#         super().__init__(**kwagrs)
#         self.num_heads = num_heads
#         self.embedding_dim = embedding_dim
#         self.attention_dropout = attention_dropout
#         self.feedforward_dropout = feedforward_dropout
#         self.norm_epsilon = norm_epsilon
#
#         self.multi_head_attention = keras.layers.MultiHeadAttention(
#             num_heads=self.num_heads,
#             key_dim=self.embedding_dim,
#             dropout=self.attention_dropout
#         )
#         self.skip_1 = keras.layers.Add()
#         self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=self.norm_epsilon)
#         self.feed_forward = FeedForward(self.embedding_dim)
#         self.skip_2 = keras.layers.Add()
#         self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=self.norm_epsilon)
#
#     def call(self, inputs):
#         attention_out = self.multi_head_attention(inputs, inputs)
#         x = self.skip_1([attention_out, inputs])
#         x = self.layer_norm_1(x)
#         feedforward_out = self.feed_forward(x)
#         x = self.skip_2([feedforward_out, x])
#         x = self.layer_norm_2(x)
#         return x



# class Encoder(layers.Layer):
#     """
#     Encoder for TabTransformer as proposed by Xin Huang et al. in the paper
#     TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
#     It simply stacks N transformer layers and applies them to the outputs
#     of the ColumnEmbedding layer to 'contextualize' them.
#
#     Reference(s):
#         https://arxiv.org/abs/2012.06678
#
#     Args:
#         num_transformer_layer: Number of transformer layers to use in the encoder
#         num_heads: Number of heads to use in the MultiHeadAttention layer
#         embedding_dim: Embedding dimensions in the MultiHeadAttention layer
#         attention_dropout: Dropout rate to use in the MultiHeadAttention layer
#         feedforward_dropout: Dropout rate to use in the FeedForward layer
#         norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
#     """
#     def __init__(self,
#                  num_transformer_layers,
#                  num_heads,
#                  embedding_dim,
#                  attention_dropout,
#                  feedforward_dropout,
#                  norm_epsilon=1e-6,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.num_transformer_layers = num_transformer_layers
#         self.num_heads = num_heads
#         self.embedding_dim = embedding_dim
#         self.attention_dropout = attention_dropout
#         self.feedforward_dropout = feedforward_dropout
#         self.norm_epsilon = norm_epsilon
#         self.transformer_layers = [Transformer(num_heads=self.num_heads,
#                                                embedding_dim=self.embedding_dim,
#                                                attention_dropout=self.attention_dropout,
#                                                feedforward_dropout=self.feedforward_dropout,
#                                                norm_epsilon=self.norm_epsilon)]
#
#     def call(self, inputs):
#         x = inputs
#         for layer in self.transformer_layers:
#             x = layer(x)
#         return x


# class RegressionHead(layers.Layer):
#     """
#     Regressor head to use on top of the TabTransformer,
#     based on the architecture proposed by Xin Huang et al. in the paper
#     TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
#
#     Reference(s):
#         https://arxiv.org/abs/2012.06678
#
#     Args:
#         units_hidden: List of units to use in hidden dense layers.
#             Number of hidden dense layers will be equal to the length of units_hidden list.
#         activation_hidden: Activation function to use in hidden dense layers.
#         units_out: Number of regression outputs.
#         use_batch_normalization: Whether to apply batch normalization after each hidden layer.
#     """
#     def __init__(self,
#                  units_hidden:list,
#                  activation_hidden="relu",
#                  num_outputs=1,
#                  use_batch_normalization=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.units_hidden = units_hidden
#         self.activation_hidden = activation_hidden
#         self.num_outputs = num_outputs
#         self.use_batch_normalization = use_batch_normalization
#         self.layers = []
#         for units in units_hidden:
#             if self.use_batch_normalization:
#                 norm = keras.layers.BatchNormalization()
#                 self.layers.append(norm)
#             dense = keras.layers.Dense(units)
#             self.layers.append(dense)
#         dense_out = keras.layers.Dense(self.num_outputs)
#         self.layers.append(dense_out)
#
#     def call(self, inputs):
#         x = inputs
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class ClassificationHead(layers.Layer):
#     """
#     Classifier head to use on top of the TabTransformer,
#     based on the architecture proposed by Xin Huang et al. in the paper
#     TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
#
#     Reference(s):
#         https://arxiv.org/abs/2012.06678
#
#     Args:
#         units_hidden: List of units to use in hidden dense layers.
#             Number of hidden dense layers will be equal to the length of units_hidden list.
#         activation_hidden: Activation function to use in hidden dense layers.
#         num_classes: Number of classes to predict.
#         activation_out: Activation to use in the output dense layer.
#             By default, sigmoid will be used for binary and softmax for multiclass classificaiton.
#         use_batch_normalization: Whether to apply batch normalization after each hidden layer.
#     """
#     def __init__(self,
#                  units_hidden:list,
#                  activation_hidden="relu",
#                  num_classes=2,
#                  activation_out=None,
#                  use_batch_normalization=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.units_hidden = units_hidden
#         self.activation_hidden = activation_hidden
#         self.num_classes = 1 if num_classes <= 2 else num_classes
#         self.activation_out = activation_out
#         if self.activation_out is None:
#             self.activation_out = "sigmoid" if self.num_classes == 1 else "softmax"
#         self.use_batch_normalization = use_batch_normalization
#         self.layers = []
#         for units in units_hidden:
#             if self.use_batch_normalization:
#                 norm = keras.layers.BatchNormalization()
#                 self.layers.append(norm)
#             dense = keras.layers.Dense(units)
#             self.layers.append(dense)
#         dense_out = keras.layers.Dense(self.num_classes,
#                                        activation=self.activation_out)
#         self.layers.append(dense_out)
#
#     def call(self, inputs):
#         x = inputs
#         for layer in self.layers:
#             x = layer(x)
#         return x