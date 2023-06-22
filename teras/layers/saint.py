import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from teras.layers.common.transformer import FeedForward, Transformer


class NumericalFeatureEmbedding(layers.Layer):
    """
    Numerical Feature Embedding layer based on the architecture proposed
    by Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        numerical_features_metadata: `dict`,
            A dictionary where for each feature in numerical features
            the feature name is mapped against its index in the dataset.
        embedding_dim: `int`, default 32,
            Embedding dimension is the dimensionality of the output layer or
            the dimensionality of the embeddings produced.
            (These embedding dimensions are the same used for the embedding categorical features)
        hidden_dim: `int`, default 16,
            Hidden dimension, used by the first dense layer i.e the hidden layer whose outputs
            are later projected to the `emebedding_dim`
    """
    def __init__(self,
                 numerical_features_metadata: dict,
                 embedding_dim: int = 32,
                 hidden_dim: int = 16
                 ):
        super().__init__()
        self.numerical_features_metadata = numerical_features_metadata
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self._num_numerical_features = len(self.numerical_features_metadata)
        # Need to create as many embedding layers as there are numerical features
        self.embedding_layers = []
        for _ in range(self._num_numerical_features):
            self.embedding_layers.append(
                models.Sequential([
                    layers.Dense(units=self.hidden_dim, activation="relu"),
                    layers.Dense(units=self.embedding_dim)
                    ]
                )
            )

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

        numerical_feature_embeddings = tf.TensorArray(size=self._num_numerical_features,
                                                      dtype=tf.float32)

        for i, (feature_name, feature_idx) in enumerate(self.numerical_features_metadata.items()):
            if self._is_data_in_dict_format:
                feature = tf.expand_dims(inputs[feature_name], 1)
            else:
                feature = tf.expand_dims(inputs[:, feature_idx], 1)
            embedding = self.embedding_layers[i]
            feature = embedding(feature)
            numerical_feature_embeddings = numerical_feature_embeddings.write(i, feature)

        numerical_feature_embeddings = tf.squeeze(numerical_feature_embeddings.stack())
        if tf.rank(numerical_feature_embeddings) == 3:
            numerical_feature_embeddings = tf.transpose(numerical_feature_embeddings, perm=[1, 0, 2])
        else:
            # else the rank must be 2
            numerical_feature_embeddings = tf.transpose(numerical_feature_embeddings)
        return numerical_feature_embeddings


class MultiHeadInterSampleAttention(layers.Layer):
    """
    MultiHeadInterSampleAttention layer as proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    Unlike the usual MultiHeadAttention layer, this MultiHeadInterSampleAttention layer,
    as the name enunciates, applies attention over samples/rows instead of features/columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_heads: `int`, default 8,
            Number of Attention heads to use
        key_dim: `int`, default 32,
            Key dimensionality for attention.
        dropout: `float`, default 0.1,
            Dropout rate to use.
    """
    def __init__(self,
                 num_heads: int = 8,
                 key_dim: int = 32,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=self.num_heads,
                                                              key_dim=self.key_dim,
                                                              dropout=dropout,
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
        embedding_dim: `int`, default 32,
            Embedding dimensions used to embedd numerical and
            categorical features. These server as the key dimensions
            in the MultiHeadAttention layer.
        num_attention_heads: `default`, default 8, Number of heads
            to use in the typical MultiHeadAttention that will be
            applied over features.
        num_inter_sample_attention_heads: `int`, default 8,
            Number of heads to use in the MultiHeadInterSampleAttention
            that will be applied over rows
        embedding_dim: `int`, default 32, Embedding dimensions. These will
            also serve as key dimensions for the MultiHeadAttention layers
        attention_dropout: `float`, default 0.1, Dropout rate for
            MultiHeadAttention which is applied over features.
        inter_sample_attention_dropout: `float`, default 0.1, Dropout rate for
            MultiHeadInterSampleAttention which is applied over rows.
        feedforward_dropout: `float`, default 0.1, Dropout rate for the
            dropout layer that is part of the FeedForward block.
        apply_attention_to_features: `bool`, default True,
            Whether to apply attention over features.
            If True, the regular MultiHeadAttention layer will be applied
            over features.
        apply_attention_to_rows: `bool`, default True,
            Whether to apply attention over rows.
            If True, the MultiHeadInterSampleAttention will apply attention
            over rows.
            NOTE: It is strongly recommended to keep both as True, but you
            can turn one off for experiment's sake.
            Also, note that, both CANNOT be False at the same time!
        num_embedded_features: `int`, Number of features that have been embedded.
            If both categorical and numerical features are embedded, then
            `num_features` is equal to the total number of features in the dataset,
            otherwise if only categorical features are embedded, then `num_features`
            is equal to the number of categorical features in the dataset.
    """
    def __init__(self,
                 embedding_dim: int = 32,
                 num_attention_heads: int = 8,
                 num_inter_sample_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 inter_sample_attention_dropout: float = 0.1,
                 feedforward_dropout: float = 0.1,
                 norm_epsilon: float = 1e-6,
                 apply_attention_to_features: bool = True,
                 apply_attention_to_rows: bool = True,
                 num_embedded_features: int = None,
                 **kwagrs):
        super().__init__(**kwagrs)
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_inter_sample_attention_heads = num_inter_sample_attention_heads
        self.attention_dropout = attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.apply_attention_to_features = apply_attention_to_features
        self.apply_attention_to_rows = apply_attention_to_rows
        self.num_embedded_features = num_embedded_features

        # We build the inner SAINT Transformer block using keras Functional API

        # Inter Sample Attention Block: this attention is applied to rows.
        inputs = layers.Input(shape=(self.num_embedded_features, self.embedding_dim))
        intermediate_outputs = inputs
        if self.apply_attention_to_rows:
            residual = inputs
            x = MultiHeadInterSampleAttention(num_heads=self.num_inter_sample_attention_heads,
                                              key_dim=self.embedding_dim * self.num_embedded_features,
                                              dropout=self.inter_sample_attention_dropout,
                                              name="inter_sample_multihead_attention"
                                              )(inputs)
            x = layers.Add()([x, residual])
            x = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)
            residual = x
            x = FeedForward(self.embedding_dim)(x)
            x = layers.Add()([x, residual])
            intermediate_outputs = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)
            final_outputs = intermediate_outputs

        # MultiHeadAttention block: this attention is applied to columns
        if self.apply_attention_to_features:
            # If `apply_attention_to_features` is set to True,
            # then attention will be applied to columns/features
            # The MultiHeadInterSampleAttention applies attention over rows,
            # but the regular MultiHeadAttention layer to apply attention over features
            # Since the common Transformer layer applies MutliHeadAttention over rows
            # as well as take care of applying all the preceding and following stuff,
            # so we'll just use that here.
            final_outputs = Transformer(embedding_dim=self.embedding_dim,
                                        num_attention_heads=self.num_attention_heads,
                                        attention_dropout=self.attention_dropout,
                                        feedforward_dropout=self.feedforward_dropout,
                                        norm_epsilon=self.norm_epsilon,
                                        name="inner_trasnformer_block_for_features")(intermediate_outputs)

        self.transformer_block = keras.Model(inputs=inputs,
                                             outputs=final_outputs,
                                             name="saint_inner_transformer_block")

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

    It differs from the typical Encoder block only in that the Transformer
    layer is a bit different from the regular Transformer layer used in the
    Transformer based architectures as it uses multi-head inter-sample attention,
    in addition to the regular mutli-head attention for features.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        num_transformer_layer: `int`, default 6,
            Number of transformer layers to use in the Encoder
        embedding_dim: `int`, default 32,
            Embedding dimensions used to embedd numerical and
            categorical features. These server as the key dimensions
            in the MultiHeadAttention layer.
        num_attention_heads: `default`, default 8, Number of heads
            to use in the typical MultiHeadAttention that will be
            applied over features.
        num_inter_sample_attention_heads: `int`, default 8,
            Number of heads to use in the MultiHeadInterSampleAttention
            that will be applied over rows
        embedding_dim: `int`, default 32, Embedding dimensions. These will
            also serve as key dimensions for the MultiHeadAttention layers
        attention_dropout: `float`, default 0.1, Dropout rate for
            MultiHeadAttention which is applied over features.
        inter_sample_attention_dropout: `float`, default 0.1, Dropout rate for
            MultiHeadInterSampleAttention which is applied over rows.
        feedforward_dropout: `float`, default 0.1, Dropout rate for the
            dropout layer that is part of the FeedForward block.
        apply_attention_to_features: `bool`, default True,
            Whether to apply attention over features.
            If True, the regular MultiHeadAttention layer will be applied
            over features.
        apply_attention_to_rows: `bool`, default True,
            Whether to apply attention over rows.
            If True, the MultiHeadInterSampleAttention will apply attention
            over rows.
            NOTE: It is strongly recommended to keep both as True, but you
            can turn one off for experiment's sake.
            Also, note that, both CANNOT be False at the same time!
        num_embedded_features: `int`, Number of features that have been embedded.
            If both categorical and numerical features are embedded, then
            `num_features` is equal to the total number of features in the dataset,
            otherwise if only categorical features are embedded, then `num_features`
            is equal to the number of categorical features in the dataset.
    """
    def __init__(self,
                 num_transformer_layers: int = 6,
                 embedding_dim: int = 32,
                 num_attention_heads: int = 8,
                 num_inter_sample_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 inter_sample_attention_dropout: float = 0.1,
                 feedforward_dropout: float = 0.1,
                 norm_epsilon: float = 1e-6,
                 apply_attention_to_features: bool = True,
                 apply_attention_to_rows: bool = True,
                 num_embedded_features: int = None,
                 **kwargs):
        super().__init__(**kwargs)

        if not apply_attention_to_features and not apply_attention_to_rows:
            raise ValueError("`apply_attention_to_features` and `apply_attention_to_rows` both cannot be False "
                             "at the same time. You must set at least one to True if not both. "
                             f"Received: `apply_attention_to_features`={apply_attention_to_features}, "
                             f"`apply_attention_to_rows`={apply_attention_to_rows}")

        self.num_transformer_layers = num_transformer_layers
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_inter_sample_attention_heads = num_inter_sample_attention_heads
        self.attention_dropout = attention_dropout
        self.inter_sample_attention_dropout = inter_sample_attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.apply_attention_to_features = apply_attention_to_features
        self.apply_attention_to_rows = apply_attention_to_rows
        self.num_embedded_features = num_embedded_features

        self.transformer_layers = keras.models.Sequential(name="transformer_layers")
        for i in range(self.num_transformer_layers):
            self.transformer_layers.add(SAINTTransformer(
                                            embedding_dim=self.embedding_dim,
                                            num_attention_heads=self.num_attention_heads,
                                            num_inter_sample_attention_heads=self.num_inter_sample_attention_heads,
                                            attention_dropout=self.attention_dropout,
                                            inter_sample_attention_dropout=self.inter_sample_attention_dropout,
                                            feedforward_dropout=self.feedforward_dropout,
                                            use_inter_sample_attention=self.use_inter_sample_attention,
                                            apply_attention_to_rows_only=self.apply_attention_to_rows_only,
                                            num_embedded_features=self.num_embedded_features,
                                            name=f"saint_transformer_layer_{i}"))

    def call(self, inputs):
        outputs = self.transformer_layers(inputs)
        return outputs
