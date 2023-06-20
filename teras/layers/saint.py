import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from teras.layers.base.transformer import FeedForward, Transformer
from teras.layers import GEGLU


# class CategoricalFeaturesEmbedding(layers.Layer):
#     """
#     Categorical Features Embedding layer based on the architecture proposed
#     by Gowthami Somepalli et al. in the paper
#     SAINT: Improved Neural Networks for Tabular Data
#     via Row Attention and Contrastive Pre-Training.
#
#     Reference(s):
#         https://arxiv.org/abs/2106.01342
#
#     Args:
#         categorical_features: List of numerical feature names.
#         lookup_tables: Look Up tables for categorical features.
#         embedding_layers: Embedding layers for categorical features.
#         """
#     def __init__(self,
#                  categorical_features,
#                  lookup_tables,
#                  embedding_layers,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.categorical_features = categorical_features
#         self.lookup_tables = lookup_tables
#         self.embedding_layers = embedding_layers
#
#     def call(self, inputs, *args, **kwargs):
#         # Encode and embedd categorical features
#         categorical_features_embeddings = []
#         for feature in self.categorical_features:
#             lookup = self.lookup_tables[feature]
#             embedding = self.embedding_layers[feature]
#             # Convert string input values to integer indices
#             encoded_feature = lookup(tf.expand_dims(inputs[feature], 1))
#             # Convert index values to embedding representations
#             encoded_feature = embedding(encoded_feature)
#             categorical_features_embeddings.append(encoded_feature)
#         return categorical_features_embeddings


class NumericalFeaturesEmbedding(layers.Layer):
    """
    Numerical Features Embedding layer based on the architecture proposed
    by Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        numerical_features: List of numerical feature names.
        hidden_dim: Hidden dimension, used by the first dense layer i.e the hidden layer
        embedding_dim: Embedding dimension, used by the second i.e. last dense layer i.e the output layer
            (these embedding dimensions are the same used for the embedding categorical features)
    """
    def __init__(self,
                 numerical_features,
                 hidden_dim,
                 embedding_dim):
        super().__init__()
        self.numerical_features = numerical_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.num_numerical_features = len(self.numerical_features)
        # Need to create as many embedding layers as there are numerical features
        self.embedding_layers = []
        for _ in range(self.num_numerical_features):
            self.embedding_layers.append(
                models.Sequential([
                    layers.Dense(units=self.hidden_dim),
                    layers.ReLU(),
                    layers.Dense(units=self.embedding_dim)
                    ]
                )
            )

    def call(self, inputs):
        x = inputs
        numerical_feature_embeddings = []
        for feature, embedding_layer in zip(self.numerical_features,
                                            self.embedding_layers):
            feature_embedding = embedding_layer(tf.expand_dims(inputs[feature], 1))
            numerical_feature_embeddings.append(feature_embedding)
        return numerical_feature_embeddings


# class FeedForward(layers.Layer):
#     """
#     FeedForward layer as used by Gowthami Somepalli et al.
#     in the paper SAINT: Improved Neural Networks for Tabular Data
#     via Row Attention and Contrastive Pre-Training.
#     The output of MutliHeadAttention layer passed through two dense layers.
#     The first layer expands the embeddings to multiplier times its size and the second layer
#     projects it back to its original size.
#
#     Reference(s):
#         https://arxiv.org/abs/2106.01342
#
#     Args:
#         input_dim: Dimensionality of the input. Here it typically means the embedding_dim.
#         multiplier: Key dimensions for MultiHeadAttention
#         dropout: Dropout rate
#         activation: Activation for the first hidden layer. Defaults to 'geglu'.
#     """
#     def __init__(self,
#                  embedding_dim,
#                  multiplier=4,
#                  dropout=0.,
#                  activation="geglu",
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.embedding_dim = embedding_dim
#         self.multiplier = multiplier
#         self.dropout = dropout
#         self.activation = GEGLU() if activation.lower() == "geglu" else activation
#
#         self.dense_1 = layers.Dense(self.embedding_dim * self.multiplier,
#                                     activation=self.activation)
#         self.dropout = layers.Dropout(self.dropout)
#         self.dense_2 = layers.Dense(self.input_dim)
#
#     def call(self, inputs):
#         x = self.dense_1(inputs)
#         x = self.dropout(x)
#         x = self.dense_2(x)
#         return x


class MultiHeadInterSampleAttention(layers.Layer):
    """
    MultiHeadInterSampleAttention layer as proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    Unlike the usual MultiHeadAttention layer, this MultiHeadInterSampleAttention layer,
    as the name enunciates, applies attention over samples or rows.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_heads: Number of Attention heads to use
        key_dim: Key dimensionality for attention
    """
    def __init__(self,
                 num_heads=None,
                 key_dim=None,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            **kwargs)

    def call(self, inputs):
        # Expected inputs shape: (b, n, d)
        # b: batch_size, n: num_features, d: embedding_dim
        x = inputs
        x = tf.reshape(x, shape=(1,
                                 tf.shape(x)[0],
                                 tf.shape(x)[1] * tf.shape(x)[2]))
        x = self.multi_head_attention(x, x)
        x = tf.reshape(x, shape=tf.shape(inputs))
        return x


# class Transformer(layers.Layer):
#     """
#     Transformer layer as used by Gowthami Somepalli et al.
#     in the paper SAINT: Improved Neural Networks for Tabular Data
#     via Row Attention and Contrastive Pre-Training.
#     It is exactly similar to the Transformer layer used in TabTransformer.
#
#     Reference(s):
#         https://arxiv.org/abs/2106.01342
#
#     Args:
#         num_heads: Number of heads to use in the MultiHeadAttention
#         key_dim: Key dimensions for MultiHeadAttention
#         attention_dropout: Dropout rate for MultiHeadAttention
#         feedforward_dropout: Dropout rate for FeedForward layer
#         norm_epsilon: Normalization value to use for Layer Normalization
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
#         self.multi_head_attention = layers.MultiHeadAttention(
#             num_heads=self.num_heads,
#             key_dim=self.embedding_dim,
#             dropout=self.attention_dropout
#         )
#         self.skip_1 = layers.Add()
#         self.layer_norm_1 = layers.LayerNormalization(epsilon=self.norm_epsilon)
#         self.feed_forward = FeedForward(self.embedding_dim)
#         self.skip_2 = layers.Add()
#         self.layer_norm_2 = layers.LayerNormalization(epsilon=self.norm_epsilon)
#
#     def call(self, inputs):
#         attention_out = self.multi_head_attention(inputs, inputs)
#         x = self.skip_1([attention_out, inputs])
#         x = self.layer_norm_1(x)
#         feedforward_out = self.feed_forward(x)
#         x = self.skip_2([feedforward_out, x])
#         x = self.layer_norm_2(x)
#         return x


class SAINTTransformer(layers.Layer):
    """
    SAINT Transformer layer as proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It differs from the usual Transformer (L) block in that it contains additional
    multihead intersample attention layer in addition to the usual multihead attention layer

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_heads_feature_attn: Number of heads to use in the
            MultiHeadAttention that will be applied over features
        num_heads_inter_sample_attn: Number of heads to use in the
            MultiHeadInterSampleAttention that will be applied over rows
        embedding_dim: Embedding dimensions. Will also serve as key dimensions
            for the multi head attention layers
        feature_attention_dropout: Dropout rate for MultiHeadAttention over features
        inter_sample_attention_dropout: Dropout rate for MultiInterSample HeadAttention over rows
        rows_only: Whether to apply attention over rows only.
        If False, attention will be apllied to both rows and columns.
        num_features: Number of features in the dataset
    """
    def __init__(self,
                 num_heads_feature_attn=None,
                 num_heads_inter_sample_attn=None,
                 embedding_dim=None,
                 feature_attention_dropout=0.,
                 inter_sample_attention_dropout=0.,
                 feedforward_dropout=0.,
                 norm_epsilon=1e-6,
                 rows_only=False,
                 num_features=None,
                 **kwagrs):
        """
        Args:
            num_heads: Number of heads to use in the MultiHeadAttention
            key_dim: Key dimensions for MultiHeadAttention
            value_dim: Value dimensions aka Embedding dimensions for MultiHeadAttention
            rows_only: If True, attention will be applied to rows only. Othewise to both rows and cols.


            # Deprecated for now. Might decide to use them in the future.
            use_intersample_attention: If True, will use Intersample attention over data points
            in addition to attention over feautures as proposed in SAINT paper (reference it properly here)

            use_intersample_attention_for_rows_only: By default it is False, intersample attention will be applied
            to rows in addition to attention being applied to features. But if set to True, only intersample attention
            will be applied and no attention will be applied to features.
        """
        super().__init__(**kwagrs)
        self.num_heads_feature_attn = num_heads_feature_attn
        self.num_heads_inter_sample_attn = num_heads_inter_sample_attn
        self.embedding_dim = embedding_dim
        self.feature_attention_dropout = feature_attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.rows_only = rows_only
        self.num_features = num_features

        # Inter Sample Attention Block
        inputs = layers.Input(shape=(num_features, embedding_dim))
        residual = inputs
        x = MultiHeadInterSampleAttention(
                                          key_dim=self.embedding_dim * self.num_features,
                                          num_heads=self.num_heads_inter_sample_attn,
                                          dropout=self.inter_sample_attention_dropout,
                                          name="inter_sample_multihead_attention"
                                          )(inputs)
        x = layers.Add()([x, residual])
        x = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)
        residual = x
        x = FeedForward(self.embedding_dim)(x)
        x = layers.Add()([x, residual])
        inter_outputs = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)
        final_outputs = inter_outputs

        # Feature (Self) Attention Block: Attention that will be applied to features
        if not rows_only:
            # if not rows only, then attention will be applied to both rows and columns/features
            residual = inter_outputs
            x = layers.MultiHeadAttention(
                num_heads=self.num_heads_feature_attn,
                key_dim=self.embedding_dim,
                dropout=self.feature_attention_dropout,
                name="features_multi_head_attention"
            )(inter_outputs, inter_outputs)
            x = layers.Add()([x, residual])
            x = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)
            residual = x
            x = FeedForward(self.embedding_dim)(x)
            x = layers.Add()([x, residual])
            final_outputs = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)

        self.transformer_block = keras.Model(inputs=inputs, outputs=final_outputs, name="saint_inner_transformer_block")

    def call(self, inputs):
        out = self.transformer_block(inputs)
        return out


class Encoder(layers.Layer):
    """
    Encoder for SAINT as proposed by Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It simply stacks N transformer layers and applies them to the outputs
    of the embedded features.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
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
        use_inter_sample_attention: Whether to use inter sample attention
        rows_only: When use_inter_sample_attention is True, this parameter determines whether to
            apply attention over just rows (when True) or over both rows and columns (when False).
            Defaults to False.
        num_features: Number of features in the input embeddings.
            (May not necessarily be equal to the number of features in the original dataset)
    """
    def __init__(self,
                 num_transformer_layers=None,
                 num_heads_feature_attn=None,
                 num_heads_inter_sample_attn=None,
                 embedding_dim=None,
                 feature_attention_dropout=0.,
                 inter_sample_attention_dropout=0.,
                 feedforward_dropout=0.,
                 norm_epsilon=1e-6,
                 use_inter_sample_attention=True,
                 apply_attention_to_rows_only=False,
                 num_features=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_transformer_layers = num_transformer_layers
        self.num_heads_feature_attn = num_heads_feature_attn
        self.num_heads_inter_sample_attn = num_heads_inter_sample_attn
        self.embedding_dim = embedding_dim
        self.feature_attention_dropout = feature_attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.use_inter_sample_attention = use_inter_sample_attention
        self.apply_attention_to_rows_only = apply_attention_to_rows_only
        self.num_features = num_features

        # TODO replace this list with a keras sequential model
        self.transformer_layers = []
        for i in range(self.num_transformer_layers):
            if self.use_inter_sample_attention:
                transformer = SAINTTransformer(num_heads_inter_sample_attn=self.num_heads_inter_sample_attn,
                                               num_heads_feature_attn=self.num_heads_feature_attn,
                                               feature_attention_dropout=self.feature_attention_dropout,
                                               inter_sample_attention_dropout=self.inter_sample_attention_dropout,
                                               embedding_dim=self.embedding_dim,
                                               feedforward_dropout=self.feedforward_dropout,
                                               rows_only=self.apply_attention_to_rows_only,
                                               num_features=self.num_features)
            else:
                transformer = Transformer(num_heads=self.num_heads_feature_attn,
                                          embedding_dim=self.embedding_dim,
                                          input_dim=self.input_dim,
                                          attention_dropout=self.feature_attention_dropout,
                                          feedforward_dropout=self.feedforward_dropout)
            self.transformer_layers.append(transformer)

    def call(self, inputs):
        x = inputs
        for layer in self.transformer_layers:
            x = layer(x)
        return x

# class RegressionHead(layers.Layer):
#     """
#     Regressor head to use on top of the SAINT model,
#     based on the architecture proposed by Gowthami Somepalli et al.
#     in the paper SAINT: Improved Neural Networks for Tabular Data
#     via Row Attention and Contrastive Pre-Training.
#
#     Reference(s):
#         https://arxiv.org/abs/2106.01342
#
#     Args:
#         units_hidden: List of units to use in hidden dense layers.
#             Number of hidden dense layers will be equal to the length of units_hidden list.
#         activation_hidden: Activation function to use in hidden dense layers.
#         units_out: Number of regression outputs.
#         use_batch_normalization: Whether to apply batch normalization after each hidden layer.
#     """
#     def __init__(self,
#                  units_hidden:list = None,
#                  activation_hidden="relu",
#                  units_out=1,
#                  use_batch_normalization=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.units_hidden = units_hidden
#         self.activation_hidden = activation_hidden
#         self.units_out = units_out
#         self.use_batch_normalization = use_batch_normalization
#         self.layers = []
#         for units in units_hidden:
#             if self.use_batch_normalization:
#                 norm = layers.BatchNormalization()
#                 self.layers.append(norm)
#             dense = layers.Dense(units, activation=activation_hidden)
#             self.layers.append(dense)
#         dense_out = layers.Dense(self.units_out)
#         self.layers.append(dense_out)
#
#     def call(self, inputs):
#         x = inputs
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class ClassificationHead(layers.Layer):
#     """
#     Classification head to use on top of the SAINT model,
#     based on the architecture proposed by Gowthami Somepalli et al.
#     in the paper SAINT: Improved Neural Networks for Tabular Data
#     via Row Attention and Contrastive Pre-Training.
#
#     Reference(s):
#         https://arxiv.org/abs/2106.01342
#
#     Args:
#         units_hidden: List of units to use in hidden dense layers.
#             Number of hidden dense layers will be equal to the length of units_hidden list.
#         activation_hidden: Activation function to use in hidden dense layers.
#         units_out: Number of regression outputs.
#         use_batch_normalization: Whether to apply batch normalization after each hidden layer.
#     """
#     def __init__(self,
#                  units_hidden:list = None,
#                  activation_hidden="relu",
#                  num_classes=2,
#                  activation_out=None,
#                  use_batch_normalization=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.units_hidden = units_hidden
#         self.activation_hidden = activation_hidden
#         self.num_classes = 1 if num_classes <= 2 else num_classes
#         self.use_batch_normalization = use_batch_normalization
#         self.activation_out = activation_out
#         if self.activation_out is None:
#             self.activation_out = 'sigmoid' if num_classes == 1 else 'softmax'
#
#         self.layers = []
#         for units in units_hidden:
#             if self.use_batch_normalization:
#                 norm = layers.BatchNormalization()
#                 self.layers.append(norm)
#             dense = layers.Dense(units, activation=activation_hidden)
#             self.layers.append(dense)
#         dense_out = layers.Dense(self.num_classes, activation=activation_out)
#         self.layers.append(dense_out)
#
#     def call(self, inputs):
#         x = inputs
#         for layer in self.layers:
#             x = layer(x)
#         return x