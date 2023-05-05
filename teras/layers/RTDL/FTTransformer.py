import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers.activations import GLU
from teras.utils import get_normalization_layer, get_activation, get_initializer
from typing import Union, List


LayerType = Union[str, keras.layers.Layer]
InitializaionType = Union[str, keras.initializers.Initializer]


class NumericalFeatureTokenizer(layers.Layer):
    """
    Numerical Feature Tokenizer as proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_features: Number of numerical features
        embedding_dim: Dimensionality of embeddings
        use_bias: Whether to use bias
        initialization: Initialization method to use for the weights and bias.
            Must be one of ["uniform", "normal"] if passed as string, otherwise
            you can pass any Keras initializer object.
    """
    def __init__(self,
                 num_features: int = None,
                 embedding_dim: int = None,
                 use_bias:  bool = True,
                 initialization: InitializaionType = "unifrom",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self._embedding_dim = embedding_dim
        self.use_bias = use_bias
        self.initialization = initialization

    def build(self, input_shape):
        initializer = get_initializer(self.initialization)

        self.W = self.add_weight(initializer=initializer, shape=(self.num_features, self._embedding_dim))
        self.b = self.add_weight(initializer=initializer, shape=(self.num_features, self._embedding_dim)) if self.use_bias else None

    @property
    def embedding_dim(self):
        return tf.shape(self.W)[1]

    def call(self, inputs):
        outputs = self.W[None] * inputs[..., None]
        # outputs = self.W * inputs
        if self.use_bias:
            outputs += self.b[None]
            # outputs += self.b
        return outputs


class CategoricalFeatureTokenizer(layers.Layer):
    """
    Categorical Feature Tokenizer as proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        categorical_features: List of categorical features/columns names in the dataset
        categorical_features_vocab: Vocabulary (dict type) of values of each categorical feature.
        embedding_dim: Embedding dimensionality.
            Official implementation uses the word embedding_dim here.
    """
    def __init__(self,
                 categorical_features: List[str],
                 categorical_features_vocab: dict,
                 embedding_dim: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_features = categorical_features
        self.categorical_features_vocab = categorical_features_vocab
        self.embedding_dim = embedding_dim
        self.lookup_tables, self.embedding_layers = self._get_lookup_tables_and_embedding_layers()
        self.concat = keras.layers.Concatenate(axis=1)

    def _get_lookup_tables_and_embedding_layers(self):
        """Lookup tables and embedding layers for each categorical feature"""
        lookup_tables = {}
        embedding_layers = {}
        for feature in self.categorical_features:
            vocab = self.categorical_features_vocab[feature]
            # Lookup Table to convert string values to integer indices
            lookup = keras.layers.StringLookup(vocabulary=vocab,
                                               mask_token=None,
                                               num_oov_indices=0,
                                               output_mode="int"
                                               )
            lookup_tables[feature] = lookup
            # Create embedding layer
            embedding = keras.layers.Embedding(input_dim=len(vocab),
                                               output_dim=self.embedding_dim)
            embedding_layers[feature] = embedding

        return lookup_tables, embedding_layers

    def call(self, inputs):
        # Encode and embedd categorical features
        categorical_features_embeddings = []
        for feature_id, feature in enumerate(self.categorical_features):
            lookup = self.lookup_tables[feature]
            embedding = self.embedding_layers[feature]
            # Convert string input values to integer indices
            feature = inputs[feature]
            encoded_feature = lookup(feature)
            # Convert index values to embedding representations
            encoded_feature = embedding(encoded_feature)
            categorical_features_embeddings.append(encoded_feature)

        categorical_features_embeddings = self.concat(categorical_features_embeddings)
        return categorical_features_embeddings


class FeatureTokenizer(layers.Layer):
    """
    Feature Tokenizer as proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.
    It combines both, categorical and numerical feature tokenizers.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        numerical_features: List of names of numerical features
        categorical_features: List of names of categorical features
        categorical_features_vocab: Vocabulary (dict type) of values of each categorical feature.
        embedding_dim: Dimensionality of tokens
        initialization: Initialization method to use for the weights and bias.
            Must be one of ["uniform", "normal"] if passed as string, otherwise
            you can pass any Keras initializer object.
    """
    def __init__(self,
                 numerical_features: list = None,
                 categorical_features: list = None,
                 categorical_features_vocab: dict = None,
                 embedding_dim: int = None,
                 initialization: InitializaionType = "uniform"):
        super().__init__()
        self.numerical_features = numerical_features
        self.num_numerical_features = len(numerical_features)
        self.categorical_features = categorical_features
        self.categorical_features_vocab = categorical_features_vocab
        self._embedding_dim = embedding_dim
        self.initialization = initialization

        self.numerical_tokenizer = NumericalFeatureTokenizer(
            num_features=self.num_numerical_features,
            embedding_dim=self._embedding_dim,
            use_bias=True,
            initialization=self.initialization
        ) if self.numerical_features else None

        self.categorical_tokenizer = CategoricalFeatureTokenizer(
            categorical_features=self.categorical_features,
            categorical_features_vocab=self.categorical_features_vocab,
            embedding_dim=self._embedding_dim
        ) if self.categorical_features else None

    @property
    def embedding_dim(self):
        return self.numerical_tokenizer.embedding_dim if self.numerical_tokenizer is None \
            else self.categorical_tokenizer.embedding_dim

    def call(self, numerical_features=None,
             categorical_features=None):
        outputs = []
        if numerical_features is not None:
            if self.numerical_tokenizer is not None:
                outputs.append(self.numerical_tokenizer(numerical_features))
            else:
                raise ValueError("Numerical Features were passed when numerical tokenizer is None"
                                 "Please initialize numerical tokenizer.")

        if categorical_features is not None:
            if self.categorical_tokenizer is not None:
                outputs.append(self.categorical_tokenizer(categorical_features))
            else:
                raise ValueError("Categorical Features were passed when categorical tokenizer is None"
                                 "Please initialize categorical tokenizer.")

        return outputs[0] if len(outputs) == 1 else tf.concat(outputs, axis=1)


class CLSToken(keras.layers.Layer):
    """
    CLS Token as proposed and implemented by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        initialization: Initialization method to use for the weights.
            Must be one of ["uniform", "normal"] if passed as string, otherwise
            you can pass any Keras initializer object.
    """
    def __init__(self,
                 embedding_dim,
                 initialization,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.initialization = initialization

    def build(self, input_shape):
        initializer = self.initialization
        if isinstance(initializer, str):
            assert self.initialization.lower() in ["uniform", "normal"], \
                ("If passed by string, only uniform and normal values are supported. "
                 "Please pass keras.initializers.<YourChoiceOfInitializer> instance"
                 " if you want to use a different initializer.")
            initializer = keras.initializers.random_uniform if self.initialization == "uniform" \
                else keras.initializers.random_normal
        initializer = keras.initializers.random_uniform if self.initialization == "uniform" else keras.initializers.random_normal
        self.weight = self.add_weight(initializer=initializer,
                                 shape=(self.embedding_dim,))

    def expand(self, *leading_dimensions: int) -> tf.Tensor:
        if not leading_dimensions:
            return self.weight
        return tf.broadcast_to(tf.expand_dims(self.weight, axis=0), (*leading_dimensions, self.weight.shape[0]))

    def call(self, inputs):
        return tf.concat([inputs, self.expand(tf.shape(inputs)[0], 1)], axis=1)


class FeedForward(layers.Layer):
    """
    FeedForward layer based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        embedding_dim: Dimensionality of tokens
        hidden_dim: Hidden dimensionality of dense layers.
        use_bias: Whether to use bias in the dense layers
        dropout_rate: Rate of dropout for the dropout layer that is
            applied in between the two dense layers
        activation: Activation function to use
    """
    def __init__(self,
                 embedding_dim: int = None,
                 hidden_dim: int = None,
                 use_bias: bool = True,
                 dropout_rate: float = 0.,
                 activation: str = "relu",
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.activation = activation

        is_glu = False
        if isinstance(self.activation, str) and self.activation == "glu":
                self.act = GLU(self.hidden_dim)
                is_glu = True
        else:
            self.act = self.activation

        self.dense_1 = layers.Dense(self.hidden_dim * 2 if is_glu else 1,
                                    use_bias=self.use_bias,
                                    activation=self.act)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.dense_2 = layers.Dense(self.embedding_dim, use_bias=self.use_bias)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


class Transformer(layers.Layer):
    """
    Transformer layer based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_heads: Number of heads to use in the MultiHeadAttention layer
        embedding_dim: Embedding dimensions in the MultiHeadAttention layer
        attention_dropout: Dropout rate to use in the MultiHeadAttention layer
        feedforward_dropout: Dropout rate to use in the FeedForward layer
        norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
    """
    def __init__(self,
                 num_heads,
                 embedding_dim,
                 attention_dropout,
                 feedforward_dropout,
                 norm_epsilon=1e-6,
                 **kwagrs):
        super().__init__(**kwagrs)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon

        self.multi_head_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embedding_dim,
            dropout=self.attention_dropout
        )
        self.skip_1 = keras.layers.Add()
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=self.norm_epsilon)
        self.feed_forward = FeedForward(self.embedding_dim)
        self.skip_2 = keras.layers.Add()
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=self.norm_epsilon)

    def call(self, inputs):
        attention_out = self.multi_head_attention(inputs, inputs)
        x = self.skip_1([attention_out, inputs])
        x = self.layer_norm_1(x)
        feedforward_out = self.feed_forward(x)
        x = self.skip_2([feedforward_out, x])
        x = self.layer_norm_2(x)
        return x


class Encoder(layers.Layer):
    """
    Encoder layer  based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_transformer_layer: Number of transformer layers to use in the encoder
        num_heads: Number of heads to use in the MultiHeadAttention layer
        embedding_dim: Embedding dimensions in the MultiHeadAttention layer
        attention_dropout: Dropout rate to use in the MultiHeadAttention layer
        feedforward_dropout: Dropout rate to use in the FeedForward layer
        norm_epsilon: Value for epsilon parameter of the LayerNormalization layer
    """
    def __init__(self,
                 num_transformer_layers,
                 num_heads,
                 embedding_dim,
                 attention_dropout,
                 feedforward_dropout,
                 norm_epsilon=1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attention_dropout = attention_dropout
        self.feedforward_dropout = feedforward_dropout
        self.norm_epsilon = norm_epsilon
        self.transformer_layers = [Transformer(num_heads=self.num_heads,
                                               embedding_dim=self.embedding_dim,
                                               attention_dropout=self.attention_dropout,
                                               feedforward_dropout=self.feedforward_dropout,
                                               norm_epsilon=self.norm_epsilon)]

    def call(self, inputs):
        x = inputs
        for layer in self.transformer_layers:
            x = layer(x)
        return x


class ClassificationHead(layers.Layer):
    """
    Classification Head layer based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        num_classes: Number of classes to predict
            applied in between the two dense layers
        normalization: Normalization layer to use before the output dense layer
        activation_out: Output activation function to use.
            By default, sigmoid will be used for binary while
            softmax will be used for multiclass classification
    """
    def __init__(self,
                 num_classes: int = None,
                 activation_out: str = None,
                 normalization: LayerType = "layer",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes= 1 if num_classes <= 2 else num_classes
        self.activation_out = activation_out
        self.normalization = normalization

        if self.activation_out is None:
            self.activation = 'sigmoid' if self.num_classes == 1 else 'softmax'

        if self.normalization is not None:
            self.norm = get_normalization_layer(normalization)
        self.dense_out = keras.layers.Dense(self.num_classes,
                                            activation=self.activation_out)

    def call(self, inputs):
        x = inputs[:, -1]
        if self.normalization is not None:
            x = self.norm(x)
        x = self.dense_out(x)
        return x


class RegressionHead(layers.Layer):
    """
    Regression Head layer based on the architecture proposed by Yury Gorishniy et al.
    in the paper Revisiting Deep Learning Models for Tabular Data
    in their FTTransformer architecture.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        units_out: Number of regression outputs
        normalization: Normalization layer to use before final dense output layer
    """
    def __init__(self,
                 units_out: int = None,
                 normalization: LayerType = "layer",
                 **kwargs):
        super().__init__(**kwargs)
        self.units_out = units_out
        self.normalization = normalization
        if self.normalization is not None:
            self.norm = get_normalization_layer(normalization)
        self.dense = keras.layers.Dense(self.units_out)

    def call(self, inputs):
        x = inputs[:, -1]
        if self.normalization is not None:
            x = self.norm(x)
        x = self.dense(x)
        return x