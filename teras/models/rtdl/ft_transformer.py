import tensorflow as tf
from tensorflow import keras
from teras.layers import (FTEncoder,
                          FTFeatureTokenizer,
                          FTCLSToken,
                          FTClassificationHead,
                          FTRegressionHead)
from typing import List, Union

LayerType = Union[str, keras.layers.Layer]


class FTTransformerClassifier(keras.Model):
    """
    FT Transformer Classifier based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        numerical_features: List of names of numerical features
        categorical_features: List of names of categorical features
        categorical_features_vocab: Vocabulary (dict type) of values of each categorical feature.
        embedding_dim: Dimensionality of tokens
        num_transformer_layer: Number of transformer layers to use in the encoder
        num_attention_heads: Number of heads to use in the MultiHeadAttention layer
        attention_dropout: Dropout rate to use in the MultiHeadAttention layer
        feedforward_dropout: Dropout rate to use in the FeedForward layer
        norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
        feedforward_hidden_dim: Hidden dimensions for the feed forward layer.
        feedforward_activation: Activation function to use in the feed forward layer.
        feedforward_normalization: Normalization layer to use in the feed forward layer.
        activation_out: Output activation function to use
        head_normlaization: Normalization layer to use in the head block
    """
    def __init__(self,
                 num_classes: int = 2,
                 numerical_features: List[str] = None,
                 categorical_features: List[str] = None,
                 categorical_features_vocab: dict = None,
                 embedding_dim: int = 32,
                 num_transformer_layers: int = 8,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 feedforward_dropout:  float = 0.05,
                 feedforward_hidden_dim: int = 32,
                 feedforward_activation: str ='relu',
                 feedforward_normalization: LayerType = 'batch',
                 activation_out: str = None,
                 head_normalization: LayerType = 'batch',
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categorical_features_vocab = categorical_features_vocab
        self.num_transformer_layers = num_transformer_layers,
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.feedforward_dropout =feedforward_dropout
        self.feedforward_activation = feedforward_activation
        self.feedforward_normalization = feedforward_normalization
        self.head_normalization = head_normalization

        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = 'sigmoid' if self.num_classes == 1 else 'softmax'

        self.num_numerical_features = len(self.numerical_features)

        self.feature_tokenizer = FTFeatureTokenizer(numerical_features=self.numerical_features,
                                                    categorical_features=self.categorical_features,
                                                    categorical_features_vocab=self.categorical_features_vocab,
                                                    embedding_dim=self.embedding_dim,
                                                    initialization="normal")
        self.cls_token = FTCLSToken(self.feature_tokenizer.embedding_dim,
                                    initialization="normal")

        self.encoder = FTEncoder(num_transformer_layers=self.num_transformer_layers,
                               num_heads=self.num_attention_heads,
                               embedding_dim=self.embedding_dim,
                               attention_dropout=self.attention_dropout,
                               feedforward_dropout=self.feedforward_dropout)
        self.head = FTClassificationHead(num_classes=self.num_classes,
                                         activation_out=self.activation_out,
                                         normalization=self.head_normalization)

    def call(self, inputs):
        numerical_input_features = None
        categorical_input_features = None
        if self.numerical_features is not None:
            numerical_input_features = tf.TensorArray(size=self.num_numerical_features,
                                                      dtype=tf.float32)
            for i, feature in enumerate(self.numerical_features):
                numerical_input_features = numerical_input_features.write(i, inputs[feature])
            numerical_input_features = tf.transpose(tf.squeeze(numerical_input_features.stack(), axis=-1))
            # numerical_input_features = np.asarray([inputs[feat] for feat in self.numerical_features])
            # numerical_input_features = numerical_input_features.squeeze().transpose()
        if self.categorical_features is not None:
            categorical_input_features = {feat: inputs[feat] for feat in self.categorical_features}
        x = self.feature_tokenizer(numerical_features=numerical_input_features,
                                   categorical_features=categorical_input_features)
        x = self.cls_token(x)
        x = self.encoder(x)
        x = self.head(x)
        return x


class FTTransformerRegressor(keras.Model):
    """
    FT Transformer Regressor based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        units_out: Number of regression outputs
        categorical_features: List of names of categorical features
        categorical_features_vocab: Vocabulary (dict type) of values of each categorical feature.
        embedding_dim: Dimensionality of tokens
        num_transformer_layer: Number of transformer layers to use in the encoder
        num_attention_heads: Number of heads to use in the MultiHeadAttention layer
        attention_dropout: Dropout rate to use in the MultiHeadAttention layer
        feedforward_dropout: Dropout rate to use in the FeedForward layer
        norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
        feedforward_hidden_dim: Hidden dimensions for the feed forward layer.
        feedforward_activation: Activation function to use in the feed forward layer.
        feedforward_normalization: Normalization layer to use in the feed forward layer.
        head_normlaization: Normalization layer to use in the head block
    """
    def __init__(self,
                 units_out: int = 1,
                 numerical_features: List[str] = None,
                 categorical_features: List[str] = None,
                 categorical_features_vocab: dict = None,
                 num_transformer_layers: int = 8,
                 embedding_dim: int = 32,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.0,
                 feedforward_hidden_dim: int = 32,
                 feedforward_dropout: float = 0.05,
                 feedforward_activation: LayerType = 'relu',
                 feedforward_normalization: LayerType = 'batch',
                 head_normlaization: LayerType = 'batch',
                 **kwargs):
        super().__init__(**kwargs)
        self.units_out = units_out
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categorical_features_vocab = categorical_features_vocab
        self.num_transformer_layers = num_transformer_layers,
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.feedforward_dropout =feedforward_dropout
        self.feedforward_activation = feedforward_activation
        self.feedforward_normalization = feedforward_normalization
        self.head_normalizaiton = head_normlaization

        self.num_numerical_features = len(self.numerical_features)

        self.feature_tokenizer = FTFeatureTokenizer(numerical_features=self.numerical_features,
                                                    categorical_features=self.categorical_features,
                                                    categorical_features_vocab=self.categorical_features_vocab,
                                                    embedding_dim=self.embedding_dim,
                                                    initialization="normal")
        self.cls_token = FTCLSToken(self.feature_tokenizer.embedding_dim,
                                    initialization="normal")

        self.encoder = FTEncoder(num_transformer_layers=self.num_transformer_layers,
                               num_heads=self.num_attention_heads,
                               embedding_dim=self.embedding_dim,
                               attention_dropout=self.attention_dropout,
                               feedforward_dropout=self.feedforward_dropout)
        self.head = FTRegressionHead(units_out=units_out,
                                     normalization=self.head_normalizaiton)

    def call(self, inputs):
        numerical_input_features = None
        categorical_input_features = None
        if self.numerical_features is not None:
            numerical_input_features = tf.TensorArray(size=self.num_numerical_features,
                                                      dtype=tf.float32)
            for i, feature in enumerate(self.numerical_features):
                numerical_input_features = numerical_input_features.write(i, tf.cast(tf.expand_dims(inputs[feature], 1), dtype=tf.float32))
            numerical_input_features = tf.transpose(tf.squeeze(numerical_input_features.stack(), axis=-1))
            # numerical_input_features = np.asarray([inputs[feat] for feat in self.numerical_features])
            # numerical_input_features = numerical_input_features.squeeze().transpose()
        if self.categorical_features is not None:
            categorical_input_features = {feat: inputs[feat] for feat in self.categorical_features}
        x = self.feature_tokenizer(numerical_features=numerical_input_features,
                                   categorical_features=categorical_input_features)
        x = self.cls_token(x)
        x = self.encoder(x)
        x = self.head(x)
        return x