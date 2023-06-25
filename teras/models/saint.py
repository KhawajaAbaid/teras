import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from warnings import warn
from teras.layers import CategoricalFeatureEmbedding
from teras.layers import SAINTNumericalFeatureEmbedding, SAINTEncoder
from teras.layers.common.transformer import (ClassificationHead,
                                             RegressionHead)
from teras.config.saint import SAINTConfig
from teras.layers.regularization import MixUp, CutMix
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
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        norm_epsilon: `float`, default 1e-6,
            A very small number used for normalization in the `LayerNormalization` layer.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
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
                 feedforward_multiplier: int = SAINTConfig.feedforward_multiplier,
                 norm_epsilon: float = SAINTConfig.norm_epsilon,
                 encode_categorical_values: bool = SAINTConfig.encode_categorical_values,
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
        self.feedforward_multiplier = feedforward_multiplier
        self.norm_epsilon = norm_epsilon
        self.encode_categorical_values = encode_categorical_values
        self.numerical_embedding_hidden_dim = numerical_embedding_hidden_dim
        self.apply_attention_to_features = apply_attention_to_features
        self.apply_attention_to_rows = apply_attention_to_rows

        self._categorical_features_metadata = self.features_metadata["categorical"]
        self._numerical_features_metadata = self.features_metadata["numerical"]
        self._num_categorical_features = len(self._categorical_features_metadata)
        self._num_numerical_features = len(self._numerical_features_metadata)
        self.num_features = self._num_numerical_features + self._num_categorical_features

        self._numerical_features_exist = self._num_numerical_features > 0
        self._categorical_features_exist = self._num_categorical_features > 0

        self._num_embedded_features = 0

        # Numerical/Continuous Features Embedding
        if self._numerical_features_exist:
            self.numerical_feature_embedding = SAINTNumericalFeatureEmbedding(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.numerical_embedding_hidden_dim,
                numerical_features_metadata=self._numerical_features_metadata
            )
            self._num_embedded_features += self._num_numerical_features

        # Categorical Features Embedding
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
                                          feedforward_multiplier=self.feedforward_multiplier,
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
        if self._categorical_features_exist:
            categorical_features = self.categorical_feature_embedding(inputs)
            features = categorical_features

        if self._numerical_features_exist:
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
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        norm_epsilon: `float`, default 1e-6,
            A very small number used for normalization in the `LayerNormalization` layer.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
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
                 feedforward_multiplier: int = SAINTConfig.feedforward_multiplier,
                 norm_epsilon: float = SAINTConfig.norm_epsilon,
                 encode_categorical_values: bool = SAINTConfig.encode_categorical_values,
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
                         feedforward_multiplier=feedforward_multiplier,
                         norm_epsilon=norm_epsilon,
                         encode_categorical_values=encode_categorical_values,
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
        feedforward_multiplier: `int`, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        norm_epsilon: `float`, default 1e-6,
            A very small number used for normalization in the `LayerNormalization` layer.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
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
                 feedforward_multiplier: int = SAINTConfig.feedforward_multiplier,
                 norm_epsilon: float = SAINTConfig.norm_epsilon,
                 encode_categorical_values: bool = SAINTConfig.encode_categorical_values,
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
                         feedforward_multiplier=feedforward_multiplier,
                         norm_epsilon=norm_epsilon,
                         encode_categorical_values=encode_categorical_values,
                         apply_attention_to_features=apply_attention_to_features,
                         apply_attention_to_rows=apply_attention_to_rows,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

        self.head = RegressionHead(num_outputs=self.num_outputs,
                                   units_values=self.head_units_values,
                                   activation_hidden="relu",
                                   normalization="batch")


class SAINTPretrainer(keras.Model):
    """
    SAINTPretrainer model based on the pretraining architecture
    for the SAINT model proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        model: `keras.Model`,
            An instance of the SAINT model that you want to pretrain.
        cutmix_probs: `float`, default 0.1,
            CutMix probability which is used in generation of mask
            that is used to mix samples together.
        mixup_alpha: `float`, default 1.0,
            Alpha value for the MixUp layer, that is used for the
            Beta distribution to sample `lambda_`
            which is used to interpolate samples.
    """
    def __int__(self,
                model: SAINT,
                cutmix_probs: float = 0.3,
                mixup_alpha: float = 1.0,
                **kwargs):
        self.model = model
        self.cutmix_probs = cutmix_probs
        self.mixup_alpha = mixup_alpha

        self.mixup = MixUp(alpha=self.alpha)
        self.cutmix = CutMix(probs=self.cutmix_probs)

        # Projection head hidden dimensions as calculated by the
        # official implementation
        projection_head_hidden_dim = 6 * self.model.embedding_dim * self.model.num_features // 5
        projection_head_output_dim = self.model.embedding_dim * self.model.num_features // 2
        self.projection_head_1 = models.Sequential(
            [
                layers.Dense(units=projection_head_hidden_dim, activation="relu"),
                layers.Dense(units=projection_head_output_dim)
            ],
            name="projection_head_for_original_data"
        )

        self.projection_head_2 = models.Sequential(
            [
                layers.Dense(units=projection_head_hidden_dim, activation="relu"),
                layers.Dense(units=projection_head_output_dim)
            ],
            name="projection_head_for_augmented_data"
        )

    def call(self, inputs):
        x = inputs

        # Apply cutmix on the raw input space
        x_prime = self.cutmix(x)

        # Embed the raw inputs as well as cutmixed data
        # TODO: This looks ugly -- maybe create a Embedding layer that wraps these two embedding layers
        p = None
        p_prime = None
        if self.model._categorical_features_exist:
            p = self.model.categorical_feature_embedding(x)
            p_prime = self.model.categorical_feature_embedding(x_prime)

        if self.model._numerical_features_exist:
            numerical_features = self.numerical_feature_embedding(x)
            numerical_features_prime = self.numerical_feature_embedding(x_prime)
            if p is not None:
                p = tf.concat([p, numerical_features],
                              axis=1)
                p_prime = tf.concat([p_prime, numerical_features_prime],
                                    axis=1)
            else:
                p = numerical_features
                p_prime = numerical_features_prime

        # Apply mixup on the embedding space -- only to the augment data
        p_prime = self.mixup(p_prime)

        # Pass these embeddings through saint encoder
        r = self.model.saint_encoder(p)
        r_prime = self.model.saint_encoder(p_prime)

        # Pass the encoded features through projection heads
        z = self.projection_head_1(r)
        z_prime = self.projection_head_2(r_prime)

        return z, z_prime
