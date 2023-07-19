import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.common.transformer import FeedForward, Transformer
from teras.layerflow.layers.saint import (SAINTTransformer as _SAINTTransformerLF,
                                          SAINTEncoder as _SAINTEncoderLF,
                                          ProjectionHead as _ProjectionHeadLF,
                                          ReconstructionBlock as _ReconstructionBlockLF,
                                          ReconstructionHead as _ReconstructionHeadLF)
from typing import Union, List, Tuple


LIST_OR_TUPLE = Union[List[int], Tuple[int]]
LAYER_OR_STR = Union[keras.layers.Layer, str]


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class NumericalFeatureEmbedding(layers.Layer):
    """
    Numerical Feature Embedding layer based on the architecture proposed
    by Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

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
                ..                                                      numerical_features,
                ..                                                      categorical_features)

        embedding_dim: ``int``, default 32,
            Embedding dimension is the dimensionality of the output layer or
            the dimensionality of the embeddings produced.
            (These embedding dimensions are the same used for the embedding categorical features)

        hidden_dim: ``int``, default 16,
            Hidden dimension, used by the first dense layer i.e the hidden layer whose outputs
            are later projected to the ``emebedding_dim``
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 hidden_dim: int = 16
                 ):
        super().__init__()
        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self._num_numerical_features = len(self.features_metadata["numerical"])
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

    def call(self, inputs):
        numerical_feature_embeddings = tf.TensorArray(size=self._num_numerical_features,
                                                      dtype=tf.float32)

        for i, (feature_name, feature_idx) in enumerate(self.features_metadata["numerical"].items()):
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
        config.update({'features_metadata': self.features_metadata,
                       'embedding_dim': self.embedding_dim,
                       'hidden_dim': self.hidden_dim,
                       })
        return config


@keras.saving.register_keras_serializable(package="teras.layers.saint")
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
        config.update({'num_heads': self.num_heads,
                       'key_dim': self.key_dim,
                       'dropout': self.dropout,
                       })
        return config


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class SAINTTransformer(_SAINTTransformerLF):
    """
    SAINT Transformer layer as proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It differs from the usual Transformer (L) block in that it contains additional
    multihead intersample attention layer in addition to the usual multihead attention layer

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the total number of features in the input dataset.

        embedding_dim: ``int``, default 32,
            Embedding dimensions used to embedd numerical and
            categorical features. These server as the key dimensions
            in the MultiHeadAttention layer.

        num_attention_heads: ``int``, default 8,
            Number of heads to use in the typical ``MultiHeadAttention``
            layer that will be applied over features.

        num_inter_sample_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadInterSampleAttention``
            that will be applied over rows

        embedding_dim: ``int``, default 32,
            Embedding dimensions. These will also serve as key dimensions
             for the attention layers

        attention_dropout: ``float``, default 0.1,
            Dropout rate for ``MultiHeadAttention`` which is applied over
             features.

        inter_sample_attention_dropout: ``float``, default 0.1,
            Dropout rate for ``MultiHeadInterSampleAttention`` which is
            applied over rows.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate for the ``Dropout`` layer that is part of the
            ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the ``embedding_dim``
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the ``FeedForward`` layer.

        apply_attention_to_features: ``bool``, default True,
            Whether to apply attention over features.
            If True, the regular ``MultiHeadAttention`` layer will be applied
            over features.

        apply_attention_to_rows: ``bool``, default True,
            Whether to apply attention over rows.
            If True, the ``MultiHeadInterSampleAttention`` will apply attention
            over rows.
            NOTE: It is strongly recommended to keep both as True, but you
            can turn one off for experiment's sake.
            Also, note that, both CANNOT be False at the same time!
    """
    def __init__(self,
                 input_dim: int,
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
                 **kwargs):
        multihead_inter_sample_attention = MultiHeadInterSampleAttention(
            num_heads=num_inter_sample_attention_heads,
            key_dim=embedding_dim * input_dim,
            dropout=inter_sample_attention_dropout,
            name="inter_sample_multihead_attention"
        )
        feed_forward = FeedForward(embedding_dim=embedding_dim,
                                   multiplier=feedforward_multiplier,
                                   dropout=feedforward_dropout)
        transformer = Transformer(embedding_dim=embedding_dim,
                                  num_attention_heads=num_attention_heads,
                                  attention_dropout=attention_dropout,
                                  feedforward_dropout=feedforward_dropout,
                                  feedforward_multiplier=feedforward_multiplier,
                                  norm_epsilon=norm_epsilon,
                                  name="inner_trasnformer_block_for_features")

        super().__init__(multihead_inter_sample_attention=multihead_inter_sample_attention,
                         feed_forward=feed_forward,
                         transformer=transformer,
                         **kwargs)

        self.input_dim = input_dim
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

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
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
        return config

    @classmethod
    def from_config(cls, config):
        # input_dim is the only positional argument
        input_dim = config.pop("input_dim")
        return cls(input_dim, **config)


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class Encoder(_SAINTEncoderLF):
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
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the total number of features in the input dataset.

        num_transformer_layer: ``int``, default 6,
            Number of transformer layers to use in the Encoder

        embedding_dim: ``int``, default 32,
            Embedding dimensions used to embedd numerical and
            categorical features. These server as the key dimensions
            in the MultiHeadAttention layer.

        num_attention_heads: ``int``, default 8,
            Number of heads to use in the typical ``MultiHeadAttention``
            layer that will be applied over features.

        num_inter_sample_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadInterSampleAttention``
            that will be applied over rows

        embedding_dim: ``int``, default 32,
            Embedding dimensions. These will also serve as key dimensions
             for the attention layers

        attention_dropout: ``float``, default 0.1,
            Dropout rate for ``MultiHeadAttention`` which is applied over
             features.

        inter_sample_attention_dropout: ``float``, default 0.1,
            Dropout rate for ``MultiHeadInterSampleAttention`` which is
            applied over rows.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate for the ``Dropout`` layer that is part of the
            ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the ``embedding_dim``
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the ``FeedForward`` layer.

        apply_attention_to_features: ``bool``, default True,
            Whether to apply attention over features.
            If True, the regular ``MultiHeadAttention`` layer will be applied
            over features.

        apply_attention_to_rows: ``bool``, default True,
            Whether to apply attention over rows.
            If True, the ``MultiHeadInterSampleAttention`` will apply attention
            over rows.
            NOTE: It is strongly recommended to keep both as True, but you
            can turn one off for experiment's sake.
            Also, note that, both CANNOT be False at the same time!
    """
    def __init__(self,
                 input_dim: int,
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
                 **kwargs):
        if not apply_attention_to_features and not apply_attention_to_rows:
            raise ValueError("`apply_attention_to_features` and `apply_attention_to_rows` both cannot be False "
                             "at the same time. You must set at least one to True if not both. "
                             f"Received: `apply_attention_to_features`={apply_attention_to_features}, "
                             f"`apply_attention_to_rows`={apply_attention_to_rows}")

        saint_transformer_layers = models.Sequential(name="saint_transformer_layers")
        for i in range(num_transformer_layers):
            saint_transformer_layers.add(SAINTTransformer(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                num_attention_heads=num_attention_heads,
                num_inter_sample_attention_heads=num_inter_sample_attention_heads,
                attention_dropout=attention_dropout,
                inter_sample_attention_dropout=inter_sample_attention_dropout,
                feedforward_dropout=feedforward_dropout,
                feedforward_multiplier=feedforward_multiplier,
                apply_attention_to_features=apply_attention_to_features,
                apply_attention_to_rows=apply_attention_to_rows,
                name=f"saint_transformer_layer_{i}"))

        super().__init__(saint_transformer_layers=saint_transformer_layers,
                         **kwargs)

        self.input_dim = input_dim
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

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'num_transformer_layers': self.num_transformer_layers,
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
                  }
        return config

    @classmethod
    def from_config(cls, config):
        # input_dim is the only positional argument
        input_dim = config.pop("input_dim")
        return cls(input_dim, **config)


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class ProjectionHead(_ProjectionHeadLF):
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
        hidden_dim: ``int``, default 64,
            Dimensionality of the hidden layer.
            In the official implementation, it is computed as follows,
            `hidden_dim = 6 * embedding_dim * number_of_features // 5`

        hidden_activation, default "relu":
            Activation function to use in the hidden layer.

        output_dim: `int`, default 32,
            Dimensionality of the output layer.
            In the official implementation, it is computed as follows,
            `output_dim = embedding_dim * number_of_features // 5`
    """
    def __init__(self,
                 hidden_dim: int = 32,
                 hidden_activation="relu",
                 output_dim: int = 8,
                 **kwargs):
        hidden_block = layers.Dense(units=hidden_dim,
                                    activation=hidden_activation)
        output_layer = layers.Dense(units=output_dim)

        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'hidden_dim': self.hidden_dim,
                  'hidden_activation': self.hidden_activation,
                  'output_dim': self.output_dim}
        return config


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class ReconstructionBlock(_ReconstructionBlockLF):
    """
    ReconstructionBlock layer that is used in constructing ReconstructionHead.
    One ReconstructionBlock is created for each feature in the dataset.
    The inputs to this layer are first mapped to hidden dimensions
    and then projected to the cardinality of the feature.

    Args:
        feature_cardinality: ``int``,
            Cardinality of the given input feature.
            For categorical features, it is equal to the number of classes
            in the feature, and for numerical features, it is equal to 1.

        hidden_dim: ``int``, default 32,
            Dimensionality of the hidden layer.

        hidden_activation: ``str``, default "relu",
            Activation function to use in the hidden layer.
    """
    def __init__(self,
                 feature_cardinality: int,
                 hidden_dim: int = 32,
                 hidden_activation="relu",
                 **kwargs):
        hidden_block = layers.Dense(hidden_dim,
                                    activation=hidden_activation,
                                    name="hidden_block")
        output_layer = layers.Dense(feature_cardinality,
                                    name="output_layer")
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)
        self.feature_cardinality = feature_cardinality
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'feature_cardinality': self.feature_cardinality,
                  'hidden_dim': self.hidden_dim,
                  'hidden_activation': self.hidden_activation,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        feature_cardinality = config.pop("feature_cardinality")
        return cls(feature_cardinality, **config)


@keras.saving.register_keras_serializable(package="teras.layers.saint")
class ReconstructionHead(_ReconstructionHeadLF):
    """
        ReconstructionHead layer for ``SAINTPretrainer`` model.
    SAINT applies a separate single hidden layer MLP block
    (here we name it, the reconstruction block)
    with an output layer where output dimensions are equal
    to the number of categories in the case of categorical
    features and 1 in the case of numerical features.

    Args:
        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
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
        embedding_dim: ``int``, default 32,
            Embedding dimensions being used in the pretraining model.
            Used in the computation of the hidden dimensions for each
            reconstruction (mlp) block for each feature.
    """
    def __init__(self,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 **kwargs):
        categorical_features_metadata = features_metadata["categorical"]
        numerical_features_metadata = features_metadata["numerical"]
        num_categorical_features = len(categorical_features_metadata)
        num_numerical_features = len(numerical_features_metadata)

        # feature_cardinalities: Dimensions of each feature in the input
        # For a categorical feature, it is equal to the number of unique categories in the feature
        # For a numerical features, it is equal to 1
        features_cardinalities = []
        # recall that categorical_features_metadata dict maps feature names to a tuple of
        # feature id and unique values in the feature
        if num_categorical_features > 0:
            features_cardinalities = list(map(lambda x: len(x[1]), categorical_features_metadata.values()))
        features_cardinalities.extend([1] * num_numerical_features)

        # For the computation of denoising loss, we use a separate MLP block for each feature
        # we call the combined blocks, reconstruction heads
        reconstruction_blocks = [
            ReconstructionBlock(feature_cardinality=cardinality,
                                hidden_dim=embedding_dim * 5,
                                )
            for cardinality in features_cardinalities]

        super().__init__(reconstruction_blocks, **kwargs)

        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'features_metadata': self.features_metadata,
                  'embedding_dim': self.embedding_dim,
                  }
        return config


