import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from teras.layers.common.transformer import FeedForward, Transformer
from teras.layers.common.head import (ClassificationHead as BaseClassificationHead,
                                      RegressionHead as BaseRegressionHead)
from typing import Union, List, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
LAYER_OR_STR = Union[keras.layers.Layer, str]


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

    def get_config(self):
        config = super().get_config()
        new_config = {'numerical_features_metadata': self.numerical_features_metadata,
                      'embedding_dim': self.embedding_dim,
                      'hidden_dim': self.hidden_dim,
                      }
        config.update(new_config)
        return config


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

    def get_config(self):
        config = super().get_config()
        new_config = {'num_heads': self.num_heads,
                      'key_dim': self.key_dim,
                      'dropout': self.dropout,
                      }
        config.update(new_config)
        return config


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
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
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
                 feedforward_multiplier: int = 4,
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
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon
        self.apply_attention_to_features = apply_attention_to_features
        self.apply_attention_to_rows = apply_attention_to_rows
        self.num_embedded_features = num_embedded_features

        # To make this layer compatible with the layerflow version of this layer,
        # we instantiate some layers as self attributes
        self.multihead_inter_sample_attention = MultiHeadInterSampleAttention(
                                                    num_heads=self.num_inter_sample_attention_heads,
                                                    key_dim=self.embedding_dim * self.num_embedded_features,
                                                    dropout=self.inter_sample_attention_dropout,
                                                    name="inter_sample_multihead_attention"
                                                    )
        self.feed_forward = FeedForward(embedding_dim=self.embedding_dim,
                                        multiplier=self.feedforward_multiplier,
                                        dropout=feedforward_dropout)
        self.transformer = Transformer(embedding_dim=self.embedding_dim,
                                       num_attention_heads=self.num_attention_heads,
                                       attention_dropout=self.attention_dropout,
                                       feedforward_dropout=self.feedforward_dropout,
                                       feedforward_multiplier=self.feedforward_multiplier,
                                       norm_epsilon=self.norm_epsilon,
                                       name="inner_trasnformer_block_for_features")

        # by default we make call to the _build_saint_inner_transformer_block function
        # when the SAINTTransformer gets constructed but when this class is used as a
        # parent to the layerflow version of this, we want to be able to reconstruct
        # the functional model using the user passed layers for attention/transformer etc.
        self._build_saint_inner_transformer_block()

    def _build_saint_inner_transformer_block(self):
        # We build the inner SAINT Transformer block using keras Functional API

        # Inter Sample Attention Block: this attention is applied to rows.
        inputs = layers.Input(shape=(self.num_embedded_features, self.embedding_dim))
        intermediate_outputs = inputs

        if self.apply_attention_to_rows:
            residual = inputs
            x = self.multihead_inter_sample_attention(inputs)
            x = layers.Add()([x, residual])
            x = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)
            residual = x
            x = self.feed_forward(x)
            x = layers.Add()([x, residual])
            intermediate_outputs = layers.LayerNormalization(epsilon=self.norm_epsilon)(x)
            final_outputs = intermediate_outputs

        # MultiHeadAttention block: this attention is applied to columns
        if self.apply_attention_to_features:
            # If `apply_attention_to_features` is set to True,
            # then attention will be applied to columns/features
            # The MultiHeadInterSampleAttention applies attention over rows,
            # but the regular MultiHeadAttention layer is used to apply attention over features.
            # Since the common Transformer layer applies MutliHeadAttention over features
            # as well as takes care of applying all the preceding and following layers,
            # so we'll just use that here.
            final_outputs = self.transformer(intermediate_outputs)

        self.transformer_block = keras.Model(inputs=inputs,
                                             outputs=final_outputs,
                                             name="saint_inner_transformer_block")

    def call(self, inputs):
        out = self.transformer_block(inputs)
        return out

    def get_config(self):
        config = super().get_config()
        new_config = {'embedding_dim': self.embedding_dim,
                      'num_attention_heads': self.num_attention_heads,
                      'num_inter_sample_attention_heads': self.num_inter_sample_attention_heads,
                      'attention_dropout': self.attention_dropout,
                      'inter_sample_attention_dropout': self.inter_sample_attention_dropout,
                      'feedforward_dropout': self.feedforward_dropout,
                      'feedforward_multiplier': self.feedforward_multiplier,
                      'norm_epsilon': self.norm_epsilon,
                      'apply_attention_to_features': self.apply_attention_to_features,
                      'apply_attention_to_rows': self.apply_attention_to_rows,
                      'num_embedded_features': self.num_embedded_features,
                      }
        config.update(new_config)
        return config


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
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
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
                 feedforward_multiplier: int = 4,
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
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon
        self.apply_attention_to_features = apply_attention_to_features
        self.apply_attention_to_rows = apply_attention_to_rows
        self.num_embedded_features = num_embedded_features

        self.saint_transformer_layers = keras.models.Sequential(name="saint_transformer_layers")
        for i in range(self.num_transformer_layers):
            self.saint_transformer_layers.add(SAINTTransformer(
                embedding_dim=self.embedding_dim,
                num_attention_heads=self.num_attention_heads,
                num_inter_sample_attention_heads=self.num_inter_sample_attention_heads,
                attention_dropout=self.attention_dropout,
                inter_sample_attention_dropout=self.inter_sample_attention_dropout,
                feedforward_dropout=self.feedforward_dropout,
                feedforward_multiplier=self.feedforward_multiplier,
                apply_attention_to_features=self.apply_attention_to_features,
                apply_attention_to_rows=self.apply_attention_to_rows,
                num_embedded_features=self.num_embedded_features,
                name=f"saint_transformer_layer_{i}"))

    def call(self, inputs):
        outputs = self.saint_transformer_layers(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'num_transformer_layers': self.num_transformer_layers,
                      'embedding_dim': self.embedding_dim,
                      'num_attention_heads': self.num_attention_heads,
                      'num_inter_sample_attention_heads': self.num_inter_sample_attention_heads,
                      'attention_dropout': self.attention_dropout,
                      'inter_sample_attention_dropout': self.inter_sample_attention_dropout,
                      'feedforward_dropout': self.feedforward_dropout,
                      'feedforward_multiplier': self.feedforward_multiplier,
                      'norm_epsilon': self.norm_epsilon,
                      'apply_attention_to_features': self.apply_attention_to_features,
                      'apply_attention_to_rows': self.apply_attention_to_rows,
                      'num_embedded_features': self.num_embedded_features,
                      }
        config.update(new_config)
        return config


class ProjectionHead(layers.Layer):
    """
    ProjectionHead layer that is used in the contrastive learning phase of
    the SAINTPretrainer to project embeddings to a lower dimension.
    According to the SAINT paper,
    "The use of a projection head to reduce dimensionality before computing
    contrastive loss is common in vision and indeed also improves results
    on tabular data."

    Reference(s):
    https://arxiv.org/abs/2106.01342

    Args:
        hidden_dim: `int`,
            Dimensionality of the hidden layer.
            In the official implementation, it is computed as follows,
            `hidden_dim = 6 * embedding_dim * number_of_features // 5`
        hidden_activation, default "relu":
            Activation function to use in the hidden layer.
        output_dim: `int`,
            Dimensionality of the output layer.
            In the official implementation, it is computed as follows,
            `output_dim = embedding_dim * number_of_featuers // 5`
    """
    def __init__(self,
                 hidden_dim: int = None,
                 hidden_activation="relu",
                 output_dim: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim

        self.hidden_block = layers.Dense(units=self.hidden_dim,
                                         activation=self.hidden_activation)
        self.output_layer = layers.Dense(units=self.output_dim)

    def call(self, inputs):
        x = self.hidden_block(inputs)
        outputs = self.output_layer(x)
        return outputs


class ReconstructionBlock(layers.Layer):
    """
    ReconstructionBlock layer that is used in constructing ReconstructionHead.
    One ReconstructionBlock is created for each feature in the dataset.

    Args:
        hidden_dim: `int`,
            Dimensionality of the hidden layer.
        hidden_activation:
            Activation function to use in the hidden layer.
        data_dim: `int`,
            Dimensionality of the given input feature.
            The inputs to this layer are first mapped to hidden dimensions
            and then projected to the dimensionality of the feature.
            For categorical features, it is equal to the number of classes
            in the feature, and for numerical feautures, it is equal to 1.
    """
    def __init__(self,
                 hidden_dim: int = None,
                 hidden_activation="relu",
                 data_dim: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_block = layers.Dense(hidden_dim,
                                         activation=hidden_activation)
        self.output_layer = layers.Dense(data_dim)

    def call(self, inputs):
        x = self.hidden_block(inputs)
        outputs = self.output_layer(x)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_dim': self.hidden_dim,
                      'hidden_activation': self.hidden_activation,
                      'data_dim': self.data_dim,
                      }
        config.update(new_config)
        return config


class ReconstructionHead(layers.Layer):
    """
    ReconstructionHead layer for SAINTPretrainer model.
    SAINT applies a separate single hidden layer MLP block
    (here we name it, the reconstruction block)
    with an output layer where output dimensions are equal
    to the number of categories in the case of categorical
    features and 1 in the case of numerical features.

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
            Embedding dimensions being used in the pretraining model.
            Used in the computation of the hidden dimensions for each
            reconstruction (mlp) block for each feature.
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 **kwargs):
        super().__init__(**kwargs)
        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim

        categorical_features_metadata = self.features_metadata["categorical"]
        numerical_features_metadata = self.features_metadata["numerical"]
        num_categorical_features = len(categorical_features_metadata)
        num_numerical_features = len(numerical_features_metadata)
        self.num_features = num_categorical_features + num_numerical_features

        # feature_dims: Dimensions of each feature in the input
        # For a categorical feature, it is equal to the number of unique categories in the feature
        # For a numerical features, it is equal to 1
        feature_dims = []
        # recall that categorical_features_metadata dict maps feature names to a tuple of
        # feature id and unique values in the feature
        if num_categorical_features > 0:
            feature_dims = list(map(lambda x: len(x[1]), categorical_features_metadata.values()))
        # for num in num_categories_per_feature:
        #     feature_dims.append(num)
        feature_dims.extend([1] * num_numerical_features)

        # For the computation of denoising loss, we use a separate MLP block for each feature
        # we call the combined blocks, reconstruction heads
        self.reconstruction_blocks = [
            ReconstructionBlock(hidden_dim=self.embedding_dim * 5,
                                data_dim=dim)
            for dim in feature_dims]

    def call(self, inputs):
        """
        Args:
            inputs: SAINT transformer outputs for the augmented data.
                Since we apply categorical and numerical embedding layers
                separately and then combine them into a new features matrix
                this effectively makes the first k features in the outputs
                categorical (since categorical embeddings are applied first)
                and all other features numerical.
                Here, k = num_categorical_features

        Returns:
            Reconstructed input features
        """
        reconstructed_inputs = []
        for idx in range(self.num_features):
            feature_encoding = inputs[:, idx]
            reconstructed_feature = self.reconstruction_blocks[idx](feature_encoding)
            reconstructed_inputs.append(reconstructed_feature)
        # the reconstructed inputs will have features equal to
        # `number of numerical features` + `number of categories in the categorical features`
        reconstructed_inputs = tf.concat(reconstructed_inputs, axis=1)
        return reconstructed_inputs

    def get_config(self):
        config = super().get_config()
        new_config = {'features_metadata': self.features_metadata,
                      'embedding_dim': self.embedding_dim,
                      }
        config.update(new_config)
        return config


class ClassificationHead(BaseClassificationHead):
    """
    Classification head for the SAINT Classifier architecture.

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        units_values: `List[int] | Tuple[int]`, default (64, 32),
            For each value in the sequence,
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the ClassificationHead.
        activation_hidden: default "relu",
            Activation function to use in hidden dense layers.
        activation_out:
            Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.
        normalization: `Layer | str`, default "batch",
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_classes: int = 2,
                 units_values: LIST_OR_TUPLE = (64, 32),
                 activation_hidden="relu",
                 activation_out=None,
                 normalization: LAYER_OR_STR = "batch",
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out,
                         normalization=normalization,
                         **kwargs)


class RegressionHead(BaseRegressionHead):
    """
    Regression head for the SAINT Regressor architecture.

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs to predict.
        units_values: `List[int] | Tuple[int]`, default (64, 32),
            For each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the RegressionHead.
        activation_hidden: default "relu",
            Activation function to use in hidden dense layers.
        normalization: `Layer | str`, default "batch",
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 units_values: LIST_OR_TUPLE = (64, 32),
                 activation_hidden="relu",
                 normalization: LAYER_OR_STR = "batch",
                 **kwargs):
        super().__init__(num_outputs=num_outputs,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         normalization=normalization,
                         **kwargs)
