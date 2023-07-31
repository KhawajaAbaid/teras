from tensorflow import keras
from teras.layers.categorical_feature_embedding import CategoricalFeatureEmbedding
from teras.layers.saint.saint_numerical_feature_embedding import SAINTNumericalFeatureEmbedding
from teras.layers.saint.saint_encoder import SAINTEncoder
from teras.layers.common.head import ClassificationHead, RegressionHead
from teras.layerflow.models.saint import (SAINT as _SAINTLF,
                                          SAINTPretrainer as _SAINTPretrainerLF)
from teras.layers.saint.saint_reconstruction_head import SAINTReconstructionHead
from teras.layers.saint.saint_projection_head import SAINTProjectionHead
from teras.layers.regularization import MixUp, CutMix
from teras.config.saint import SAINTConfig
from teras.utils.types import UnitsValuesType


@keras.saving.register_keras_serializable("teras.models")
class SAINT(_SAINTLF):
    """
    SAINT architecture proposed by Gowthami Somepalli et al.
    in the paper,
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.

    SAINT performs attention over both rows and columns.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        features_metadata: ``dict``,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Embedding dimensions used in embedding numerical and categorical features.

        numerical_embedding_hidden_dim: ``int`` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            ``SAINTNumericalFeatureEmebedding`` layer.

        num_transformer_layer: ``int``, default 6,
            Number of (SAINT) transformer layers to use in the Encoder.
            The encoder is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.

        num_inter_sample_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadInterSampleAttention`` that applies
            attention over rows.

        attention_dropout: ``float``, default 0.1, Dropout rate to use in the
            ``MultiHeadSelfAttention`` layer in the transformer layer.

        inter_sample_attention_dropout: `float`, default 0.1,
            Dropout rate for ``MultiHeadInterSampleAttention`` layer that applies
            attention over rows.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate to use for the dropout layer in the ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.

        norm_epsilon: ``float``, default 1e-6,
            A very small number used for normalization in the ``LayerNormalization`` layer.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.

        apply_attention_to_features: ``bool``, default True,
            Whether to apply attention over features using the regular ``MultiHeadAttenion`` layer.

        apply_attention_to_rows: ``bool``, default True,
            Whether to apply attention over rows using the SAINT ``MultiHeadInterSampleAttention``
            layer.
            Although it is strongly recommended to apply attention to both rows and features,
            but for experimentation's sake you can disable one of them, but NOT both at the
            same time!
    """
    def __init__(self,
                 input_dim: int,
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
        num_categorical_features = len(features_metadata["categorical"])
        num_numerical_features = len(features_metadata["numerical"])
        numerical_features_exist = num_numerical_features > 0
        categorical_features_exist = num_categorical_features > 0

        # Numerical/Continuous Features Embedding
        numerical_feature_embedding = None
        if numerical_features_exist:
            # If numerical features exist, then they must be embedded
            numerical_feature_embedding = SAINTNumericalFeatureEmbedding(
                features_metadata=features_metadata,
                embedding_dim=embedding_dim,
                hidden_dim=numerical_embedding_hidden_dim
            )

        # Categorical Features Embedding
        categorical_feature_embedding = None
        if categorical_features_exist:
            # If categorical features exist, then they must be embedded
            categorical_feature_embedding = CategoricalFeatureEmbedding(
                features_metadata=features_metadata,
                embedding_dim=embedding_dim,
                encode=encode_categorical_values)

        encoder = SAINTEncoder(data_dim=input_dim,
                               num_transformer_layers=num_transformer_layers,
                               embedding_dim=embedding_dim,
                               num_attention_heads=num_attention_heads,
                               num_inter_sample_attention_heads=num_inter_sample_attention_heads,
                               attention_dropout=attention_dropout,
                               inter_sample_attention_dropout=inter_sample_attention_dropout,
                               feedforward_dropout=feedforward_dropout,
                               feedforward_multiplier=feedforward_multiplier,
                               norm_epsilon=norm_epsilon,
                               apply_attention_to_features=apply_attention_to_features,
                               apply_attention_to_rows=apply_attention_to_rows,
                               )

        super().__init__(input_dim=input_dim,
                         encoder=encoder,
                         categorical_feature_embedding=categorical_feature_embedding,
                         numerical_feature_embedding=numerical_feature_embedding,
                         **kwargs)

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

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'features_metadata': self.features_metadata,
                  'embedding_dim': self.embedding_dim,
                  'numerical_embedding_hidden_dim': self.numerical_embedding_hidden_dim,
                  'num_transformer_layers': self.num_transformer_layers,
                  'num_attention_heads': self.num_attention_heads,
                  'num_inter_sample_attention_heads': self.num_inter_sample_attention_heads,
                  'attention_dropout': self.attention_dropout,
                  'inter_sample_attention_dropout': self.inter_sample_attention_dropout,
                  'feedforward_dropout': self.feedforward_dropout,
                  'feedforward_multiplier': self.feedforward_multiplier,
                  'norm_epsilon': self.norm_epsilon,
                  'encode_categorical_values': self.encode_categorical_values,
                  'apply_attention_to_features': self.apply_attention_to_features,
                  'apply_attention_to_rows': self.apply_attention_to_rows,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        features_metadata = config.pop("features_metadata")
        return cls(input_dim=input_dim,
                   features_metadata=features_metadata, **config)


@keras.saving.register_keras_serializable("teras.models")
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
        num_classes: ``int``, default 2,
            Number of classes to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default [64, 32],
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        features_metadata: ``dict``,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Embedding dimensions used in embedding numerical and categorical features.

        numerical_embedding_hidden_dim: ``int`` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            ``SAINTNumericalFeatureEmebedding`` layer.

        num_transformer_layer: ``int``, default 6,
            Number of (SAINT) transformer layers to use in the Encoder.
            The encoder is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.

        num_inter_sample_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadInterSampleAttention`` that applies
            attention over rows.

        attention_dropout: ``float``, default 0.1, Dropout rate to use in the
            ``MultiHeadSelfAttention`` layer in the transformer layer.

        inter_sample_attention_dropout: `float`, default 0.1,
            Dropout rate for ``MultiHeadInterSampleAttention`` layer that applies
            attention over rows.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate to use for the dropout layer in the ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.

        norm_epsilon: ``float``, default 1e-6,
            A very small number used for normalization in the ``LayerNormalization`` layer.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.

        apply_attention_to_features: ``bool``, default True,
            Whether to apply attention over features using the regular ``MultiHeadAttenion`` layer.

        apply_attention_to_rows: ``bool``, default True,
            Whether to apply attention over rows using the SAINT ``MultiHeadInterSampleAttention``
            layer.
            Although it is strongly recommended to apply attention to both rows and features,
            but for experimentation's sake you can disable one of them, but NOT both at the
            same time!
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: UnitsValuesType = (64, 32),
                 input_dim: int = None,
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
        if features_metadata is None:
            raise ValueError("`features_metadata` cannot be None. "
                             "You can get features metadata dictionary using the `get_features_metadata_for_embedding` "
                             "function from teras.utils.")
        # Since the parent `SAINT` class subclasses its LayerFlow version
        # which accepts layers as input and does expose a head argument to pass the layer.
        # we can create a relevant head layer, in this case, the `ClassificationHead` layer
        # and pass its instance to the parent who will pass it along to its parent as part of
        # the **kwargs dict
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_values,
                                  activation_hidden="relu",
                                  normalization="batch",
                                  name="saint_classifier_head")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
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
                         head=head,
                         **kwargs)
        self.num_classes = num_classes
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'head_units_values': self.head_units_values
                       })
        return config


@keras.saving.register_keras_serializable("teras.models")
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
        num_outputs: ``int``, default 1,
            Number of regression outputs to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default [64, 32],
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        features_metadata: ``dict``,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Embedding dimensions used in embedding numerical and categorical features.

        numerical_embedding_hidden_dim: ``int`` default 16,
            Dimensionality of the hidden layer that precedes the output layer in the
            ``SAINTNumericalFeatureEmebedding`` layer.

        num_transformer_layer: ``int``, default 6,
            Number of (SAINT) transformer layers to use in the Encoder.
            The encoder is used to contextualize the learned feature embeddings.

        num_attention_heads: ``int``, default 8,
            Number of attention heads to use in the ``MultiHeadSelfAttention`` layer
            that is part of the `Transformer` layer which in turn is part of the `Encoder`.

        num_inter_sample_attention_heads: ``int``, default 8,
            Number of heads to use in the ``MultiHeadInterSampleAttention`` that applies
            attention over rows.

        attention_dropout: ``float``, default 0.1, Dropout rate to use in the
            ``MultiHeadSelfAttention`` layer in the transformer layer.

        inter_sample_attention_dropout: `float`, default 0.1,
            Dropout rate for ``MultiHeadInterSampleAttention`` layer that applies
            attention over rows.

        feedforward_dropout: ``float``, default 0.1,
            Dropout rate to use for the dropout layer in the ``FeedForward`` layer.

        feedforward_multiplier: ``int``, default 4.
            Multiplier that is multiplied with the `embedding_dim`
            and the resultant value is used as hidden dimensions value for the
            hidden layer in the feedforward block.

        norm_epsilon: ``float``, default 1e-6,
            A very small number used for normalization in the ``LayerNormalization`` layer.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values.
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.

        apply_attention_to_features: ``bool``, default True,
            Whether to apply attention over features using the regular ``MultiHeadAttenion`` layer.

        apply_attention_to_rows: ``bool``, default True,
            Whether to apply attention over rows using the SAINT ``MultiHeadInterSampleAttention``
            layer.
            Although it is strongly recommended to apply attention to both rows and features,
            but for experimentation's sake you can disable one of them, but NOT both at the
            same time!
    """

    def __init__(self,
                 num_outputs: int = 1,
                 head_units_values: UnitsValuesType = (64, 32),
                 input_dim: int = None,
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
        if features_metadata is None:
            raise ValueError("`features_metadata` cannot be None. "
                             "You can get features metadata dictionary using the `get_features_metadata_for_embedding` "
                             "function from teras.utils.")
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_values,
                              activation_hidden="relu",
                              normalization="batch",
                              name="saint_regressor_head")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
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
                         head=head,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_outputs': self.num_outputs,
                       'head_units_values': self.head_units_values,
                       })
        return config


@keras.saving.register_keras_serializable("teras.models")
class SAINTPretrainer(_SAINTPretrainerLF):
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
        model: ``SAINT``,
            An instance of the ``SAINT`` model that you want to pretrain.
            Note that, you should use the base ``SAINT`` model's instance,
            not ``SAINTClassifier`` or ``SAINTRegressor``.
            Using default API, you can import it as,
                >>> from teras.models import SAINT
            Using LayerFlow API, you can import it as,
                >>> from teras.layerflow.models import SAINT
                And REMEMBER to leave the ``head`` argment as None.

        features_metadata: ``dict``,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        embedding_dim: ``int``, default 32,
            Embedding dimensions being used by the base model for embedding
            numerical and categorical features.

        cutmix_probs: ``float``, default 0.1,
            ``CutMix`` probability which is used in generation of mask
            that is used to mix samples together.

        mixup_alpha: ``float``, default 1.0,
            Alpha value for the ``MixUp`` layer, that is used for the
            Beta distribution to sample `lambda_`
            which is used to interpolate samples.

        temperature: ``float``, default 0.7,
            Temperature value used in the computation of the InfoNCE contrastive loss.

        lambda_: ``float``, default 10,
            Controls the weightage of denoising loss in the summation of denoising and
            contrastive loss.
    """
    def __init__(self,
                 model: SAINT,
                 features_metadata: dict,
                 embedding_dim: int = 32,
                 cutmix_probs: float = 0.3,
                 mixup_alpha: float = 1.0,
                 temperature: float = 0.7,
                 lambda_: float = 10.,
                 **kwargs):
        mixup = MixUp(alpha=mixup_alpha)
        cutmix = CutMix(probs=cutmix_probs)
        num_features = len(features_metadata["categorical"]) + len(features_metadata["numerical"])

        # For the computation of contrastive loss, we use projection heads.
        # Projection head hidden dimensions as calculated by the
        # official implementation
        projection_head_hidden_dim = 6 * embedding_dim * num_features // 5
        projection_head_output_dim = embedding_dim * num_features // 2

        projection_head_1 = SAINTProjectionHead(hidden_dim=projection_head_hidden_dim,
                                                output_dim=projection_head_output_dim,
                                                name="projection_head_for_original_data")

        projection_head_2 = SAINTProjectionHead(hidden_dim=projection_head_hidden_dim,
                                                output_dim=projection_head_output_dim,
                                                name="projection_head_for_augmented_data")

        reconstruction_head = SAINTReconstructionHead(features_metadata=model.features_metadata,
                                                      embedding_dim=model.embedding_dim)

        super().__init__(model=model,
                         features_metadata=features_metadata,
                         mixup=mixup,
                         cutmix=cutmix,
                         projection_head_1=projection_head_1,
                         projection_head_2=projection_head_2,
                         reconstruction_head=reconstruction_head,
                         **kwargs)
        self.model = model
        self.features_metadata = features_metadata
        self.embedding_dim = embedding_dim
        self.cutmix_probs = cutmix_probs
        self.mixup_alpha = mixup_alpha
        self.temperature = temperature
        self.lambda_ = lambda_

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'model': keras.layers.serialize(self.model),
                  'features_metadata': self.features_metadata,
                  'embedding_dim': self.embedding_dim,
                  'cutmix_probs': self.cutmix_probs,
                  'mixup_alpha': self.mixup_alpha,
                  'temperature': self.temperature,
                  'lambda_': self.lambda_,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        features_metadata = config.pop("features_metadata")
        return cls(model, features_metadata, **config)
