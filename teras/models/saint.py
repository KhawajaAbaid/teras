import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers import CategoricalFeatureEmbedding
from teras.layers import SAINTNumericalFeatureEmbedding, SAINTEncoder
from teras.layers.saint import ReconstructionHead, ClassificationHead, RegressionHead, ProjectionHead
from teras.config.saint import SAINTConfig
from teras.layers.regularization import MixUp, CutMix
from teras.layers.encoding import LabelEncoding
from teras.losses.saint import info_nce_loss, denoising_loss
from teras.utils import convert_dict_to_array_tensor
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

    def get_config(self):
        config = super().get_config()
        new_config = {'features_metadata': self.features_metadata,
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
        config.update(new_config)
        return config


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
                                       activation_out=self.activation_out,
                                       name="saint_classification_head")

    def get_config(self):
        config = super().get_config()
        new_config = {'num_classes': self.num_classes,
                      'activation_out': self.activation_out
                      }
        config.update(new_config)
        return config


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
                                   name="saint_regression_head")

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      }
        config.update(new_config)
        return config


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
        temperature: `float`, default 0.7,
            Temperature value used in the computation of the InfoNCE contrastive loss.
        lambda_: `float`, default 10,
            Controls the weightage of denoising loss in the summation of denoising and
            contrastive loss.
    """
    def __init__(self,
                 model: SAINT,
                 cutmix_probs: float = 0.3,
                 mixup_alpha: float = 1.0,
                 temperature: float = 0.7,
                 lambda_: float = 10.,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.cutmix_probs = cutmix_probs
        self.mixup_alpha = mixup_alpha
        self.temperature = temperature
        self.lambda_ = lambda_

        self.mixup = MixUp(alpha=self.mixup_alpha)
        self.cutmix = CutMix(probs=self.cutmix_probs)

        # For the computation of contrastive loss, we use projection heads.
        # Projection head hidden dimensions as calculated by the
        # official implementation
        projection_head_hidden_dim = 6 * self.model.embedding_dim * self.model.num_features // 5
        projection_head_output_dim = self.model.embedding_dim * self.model.num_features // 2

        self.projection_head_1 = ProjectionHead(hidden_dim=projection_head_hidden_dim,
                                                output_dim=projection_head_output_dim,
                                                name="projection_head_for_original_data")

        self.projection_head_2 = ProjectionHead(hidden_dim=projection_head_hidden_dim,
                                                output_dim=projection_head_output_dim,
                                                name="projection_head_for_augmented_data")

        self.reconstruction_head = ReconstructionHead(features_metadata=self.model.features_metadata,
                                                      embedding_dim=self.model.embedding_dim)
        self.contrastive_loss_tracker = keras.metrics.Mean(name="contrastive_loss")
        self.denoising_loss_tracker = keras.metrics.Mean(name="denoising_loss")

        # We set concatenate_numerical_features because we want the layer to return the whole data including numerical
        # features and not just categorical features
        # NOTE: we must set keep_features_order=True if there are layers like CategoricalFeatureEmbedding that depened
        # heavily on the feature indices in case of array format input and the LabelEncoding returns data in array format
        self.label_encoding = LabelEncoding(categorical_features_metadata=self.model.features_metadata["categorical"],
                                            concatenate_numerical_features=True,
                                            keep_features_order=True)

        self._is_first_batch = True

    def get_pretrained_model(self):
        """Returns pretrained model"""
        return self.model

    @property
    def pretrained_model(self):
        """Returns pretrained model"""
        return self.model

    def compile(self,
                contrastive_loss=info_nce_loss,
                denoising_loss=denoising_loss,
                **kwargs):
        super().compile(**kwargs)
        self.contrastive_loss = contrastive_loss
        self.denoising_loss = denoising_loss

    def call(self, inputs):
        # Okay this is going to be a bit ugly solution but bear with me
        # Because we receive a model instance and not create an instance ourselves,
        # and becuse the encoding is currently merged in the emebdding layer, we have two options
        # 1. Create a separate copy of inputs, encode it and pass it to the layers like cutmix
        #   which cannot handle encoding.
        # 2. Set the model's categorical_feature_embedding layer's encode attribute to False
        #   so it doesn't encode values when called.

        # Since we have to set the encode attribute back to True if it was True originally
        # and we have to do it at the last batch of last epoch, which we cannot know,
        # so we essentially will set it to False at the start of this call method
        # and back to True at the end.
        self.model.categorical_feature_embedding.encode = False
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
            numerical_features = self.model.numerical_feature_embedding(x)
            numerical_features_prime = self.model.numerical_feature_embedding(x_prime)
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
        # Normalize
        z = z / tf.norm(z, axis=-1, keepdims=True)
        z_prime = z_prime / tf.norm(z_prime, axis=-1, keepdims=True)
        # Flatten last two dimensions
        z = tf.reshape(z, shape=(tf.shape(z)[0], tf.reduce_prod(tf.shape(z)[1:])))
        z_prime = tf.reshape(z_prime, shape=(tf.shape(z_prime)[0], tf.reduce_prod(tf.shape(z_prime)[1:])))

        # To reconstruct the input data, we pass the encodings of augmented data
        # i.e. the `r_prime` through a reconstruction head
        reconstructed_samples = self.reconstruction_head(r_prime)

        # Since we want to keep the encode value to what originally was,
        # which is also stored in the model's encode_categorical_values attribute, so we set it equal to that
        self.model.categorical_feature_embedding.encode = self.model.encode_categorical_values
        return z, z_prime, reconstructed_samples

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # At each batch, if the user has set the encode categorical values flag to True,
        # we encode the categorical (string) values in the data to make life easier and efficient
        # down the road.
        if self.model.encode_categorical_values:
            data = self.label_encoding(data)

        elif isinstance(data, dict):
            # If encode flag is set to false, then that means that all values are numerical
            # and hence the data is in homogenous form and hence we can convert data to
            # array format and save us lots of trouble down the road.
            # Sure, teras's feature embedding layers can handle dict type data but layers
            # like cutmix where we have to shuffle the data will be a pain to work with dicts.
            # BTW, the label encoding layer above not only encodes string values but also
            # converts dictionary data to array format but in case user doesn't want to encode
            # but has data in dict format, we convert it to array format.
            # Basically what we want to do is to make it so data from this point onwards is in
            # array format.
            data = convert_dict_to_array_tensor(data)

        if self._is_first_batch:
            if self.model.head is not None:
                dummy_inputs = tf.zeros(tf.shape(data))
                # since we don't need the head during pretraining
                # but not creating its weights causes trouble, so we call it on dummy
                # inputs to just initialize the weights on the first batch.
                self.model.head(dummy_inputs)
            self._is_first_batch = False

        with tf.GradientTape() as tape:
            z, z_prime, reconstructed_samples = self(data)
            c_loss = self.contrastive_loss(real_projection_outputs=z,
                                           augmented_projection_outputs=z_prime,
                                           temperature=self.temperature)
            d_loss = self.denoising_loss(real_samples=data,
                                         reconstructed_samples=reconstructed_samples,
                                         categorical_features_metadata=self.model._categorical_features_metadata)

            loss = c_loss + self.lambda_ * d_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.contrastive_loss_tracker.update_state(c_loss)
        self.denoising_loss_tracker.update_state(d_loss)
        # If user has passed any additional metrics to compile, we should update their states
        if len(self.compiled_metrics.metrics) > 0:
            self.compiled_metrics.update_state(data, reconstructed_samples)
        # If user has passed any additional losses to compile, we should call them
        if self.compiled_loss._losses is not None:
            self.compiled_loss(data, reconstructed_samples)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def get_config(self):
        config = super().get_config()
        new_config = {'model': keras.layers.serialize(self.model),
                      'cutmix_probs': self.cutmix_probs,
                      'mixup_alpha': self.mixup_alpha,
                      'temperature': self.temperature,
                      'lambda_': self.lambda_,
                      }
        config.update(new_config)
        return config
