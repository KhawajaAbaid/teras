import tensorflow as tf
from tensorflow import keras
from teras.layers import FTNumericalFeatureEmbedding, FTCLSToken
from teras.layers.embedding import CategoricalFeatureEmbedding
from teras.layers.common.transformer import Encoder, ClassificationHead, RegressionHead
from typing import List, Union
from warnings import warn


LayerType = Union[str, keras.layers.Layer]


class FTTransformer(keras.Model):
    """
    FT Transformer architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

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
            Multiplier that is multipled with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
        embed_numerical_features: `bool`, default True,
            Whether to embed the numerical features.
            If False, FTTransformer's  `FTNumericalFeatureEmbedding` layer won't be applied to
            numerical features instead they will just be fed to the encoder in raw form.
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 8,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 feedforward_dropout:  float = 0.05,
                 feedforward_multiplier: int = 4,
                 encode_categorical_values: bool = True,
                 embed_numerical_features: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.features_metadata = features_metadata
        self.num_transformer_layers = num_transformer_layers
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.feedforward_multiplier = feedforward_multiplier
        self.encode_categorical_values = encode_categorical_values
        self.embed_numerical_features = embed_numerical_features

        self._categorical_features_metadata = self.features_metadata["categorical"]
        self._numerical_features_metadata = self.features_metadata["numerical"]
        self._num_categorical_features = len(self._categorical_features_metadata)
        self._num_numerical_features = len(self._numerical_features_metadata)

        self._numerical_features_exists = self._num_numerical_features > 0
        self._categorical_features_exist = self._num_categorical_features > 0

        # Numerical/Continuous Features Embedding
        self.numerical_feature_embedding = None
        if self.embed_numerical_features:
            if self._numerical_features_exists:
                self.numerical_feature_embedding = FTNumericalFeatureEmbedding(
                                                        numerical_features_metadata=self._numerical_features_metadata,
                                                        embedding_dim=self.embedding_dim)
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

        self.cls_token = FTCLSToken(self.embedding_dim,
                                    initialization="normal")
        self.encoder = Encoder(num_transformer_layers=self.num_transformer_layers,
                               num_heads=self.num_attention_heads,
                               embedding_dim=self.embedding_dim,
                               attention_dropout=self.attention_dropout,
                               feedforward_dropout=self.feedforward_dropout,
                               feedforward_multiplier=self.feedforward_multiplier)

        self.head = None

    def call(self, inputs):
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

        elif self._numerical_features_exists:
            # Then it means that the embed numerical features is set to False
            # but numerical features do exist in the dataset, so,
            # we feed raw numerical features to the encoder
            numerical_features = tf.TensorArray(size=self._num_numerical_features,
                                                dtype=tf.float32)
            for i, (feature_name, feature_idx) in enumerate(self._numerical_features_metadata.items()):
                if self._is_data_in_dict_format:
                    feature = tf.expand_dims(inputs[feature_name], axis=1)
                else:
                    feature = tf.expand_dims(inputs[:, feature_idx], axis=1)
                numerical_features = numerical_features.write(i, feature)
            numerical_features = tf.transpose(tf.squeeze(numerical_features.stack()))
            if features is not None:
                features = tf.concat([features, numerical_features],
                                     axis=1)
            else:
                features = numerical_features

        features_with_cls_token = self.cls_token(features)
        outputs = self.encoder(features_with_cls_token)
        if self.head is not None:
            # Since FTTransformer employs BERT like CLS token.
            # So, it makes its predictions
            # using only the CLS token, not the entire dataset.
            outputs = self.head(outputs[:, -1])
        return outputs


class FTTransformerClassifier(FTTransformer):
    """
    FTTransformerClassifier based on the FTTransformer architecture proposed
    by Yury Gorishniy et al. in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
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
            Multiplier that is multipled with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
        embed_numerical_features: `bool`, default True,
            Whether to embed the numerical features.
            If False, FTTransformer's  `FTNumericalFeatureEmbedding` layer won't be applied to
            numerical features instead they will just be fed to the encoder in raw form.
    """
    def __init__(self,
                 num_classes: int = 2,
                 activation_out=None,
                 features_metadata: dict = None,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 8,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 feedforward_dropout:  float = 0.05,
                 feedforward_multiplier: int = 4,
                 encode_categorical_values: bool = True,
                 embed_numerical_features: bool = True,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         encode_categorical_values=encode_categorical_values,
                         embed_numerical_features=embed_numerical_features,
                         **kwargs)
        self.num_classes = num_classes
        self.activation_out = activation_out
        self.head = ClassificationHead(num_classes=self.num_classes,
                                       units_values=None,
                                       activation_out=self.activation_out,
                                       normalization="layer")


class FTTransformerRegressor(FTTransformer):
    """
    FTTransformerRegressor based on the FTTransformer architecture proposed
    by Yury Gorishniy et al. in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_outputs: `int`, default 1,
            Number of outputs to predict.
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
            Multiplier that is multipled with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.
        encode_categorical_values: `bool`, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
        embed_numerical_features: `bool`, default True,
            Whether to embed the numerical features.
            If False, FTTransformer's  `FTNumericalFeatureEmbedding` layer won't be applied to
            numerical features instead they will just be fed to the encoder in raw form.
    """

    def __init__(self,
                 num_outputs: int = 1,
                 features_metadata: dict = None,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 8,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 feedforward_dropout: float = 0.05,
                 feedforward_multiplier: int = 4,
                 encode_categorical_values: bool = True,
                 embed_numerical_features: bool = True,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         embedding_dim=embedding_dim,
                         num_transformer_layers=num_transformer_layers,
                         num_attention_heads=num_attention_heads,
                         attention_dropout=attention_dropout,
                         feedforward_dropout=feedforward_dropout,
                         feedforward_multiplier=feedforward_multiplier,
                         encode_categorical_values=encode_categorical_values,
                         embed_numerical_features=embed_numerical_features,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head = RegressionHead(num_outputs=self.num_outputs,
                                   units_values=None,
                                   normalization="layer")