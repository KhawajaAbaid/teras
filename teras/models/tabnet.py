from tensorflow import keras
from teras.layers.tabnet.tabnet_encoder import TabNetEncoder
from teras.layers.tabnet.tabnet_decoder import TabNetDecoder
from teras.layers.categorical_feature_embedding import CategoricalFeatureEmbedding
from teras.layers.common.head import (ClassificationHead,
                                      RegressionHead)
from teras.config.tabnet import TabNetConfig
from teras.layerflow.models.tabnet import (TabNet as _TabNetLF,
                                           TabNetPretrainer as _TabNetPretrainerLF)
from teras.utils.types import UnitsValuesType


@keras.saving.register_keras_serializable(package="keras.models")
class TabNet(_TabNetLF):
    """
    TabNet model as proposed by Sercan et al. in TabNet paper.
    This purpose will serve as the parent class for the TabNetRegressor and TabNetClassifier.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the input dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            ``{feature_name: (feature_idx, vocabulary)}`` for feature in categorical features.
            ``{feature_name: feature_idx}`` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        feature_transformer_dim: ``int``, default 32,
            The dimensionality of the hidden representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.

        decision_step_output_dim: ``int``, default 32,
            The dimensionality of output at each decision step, which is later mapped to the
            final classification or regression output.
            It is recommended to keep ``decision_step_output_dim`` and ``feature_transformer_dim``
            equal to each other.
            Adjusting these two parameters values is a good way of obtaining a tradeoff between
            performance and complexity.

        num_decision_steps: ``int``, default 5,
            The number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset, the value tends to
            be higher. That said, a very high value of `num_decision_steps` may suffer
            from overfitting.

        num_shared_layers: ``int``, default 2,
            Number of shared layers to use in the ``TabNetFeatureTransformer``.
            These shared layers are *shared* across decision steps.

        num_decision_dependent_layers: ``int``, default 2,
            Number of decision dependent layers to use in the ``TabNetFeatureTransformer``.
            In simple words, ``num_decision_dependent_layers`` are created
            for each decision step in the ``num_decision_steps``.
            For instance, if ``num_decision_steps = `5` and  ``num_decision_dependent_layers = 2``
            then 10 layers will be created, 2 for each decision step.

        relaxation_factor: ``float``, default 1.5,
            Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger ``relaxation_factor``.

        batch_momentum: ``float``, default 0.9,
            Momentum value to use for ``BatchNormalization`` layer.

        virtual_batch_size: `int`, default 64,
            Batch size to use for ``virtual_batch_size`` parameter in ``BatchNormalization`` layer.
            This is typically much smaller than the ``batch_size`` used for training.

        residual_normalization_factor: ``float``, default 0.5,
            In the feature transformer, except for the first layer, every other layer utilizes
            normalized residuals, where ``residual_normalization_factor``
            determines the scale of normalization.

        epsilon: ``float``, default 0.00001,
            Epsilon is a small number for numerical stability
            during the computation of entropy loss.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.
    """
    def __init__(self,
                 input_dim: int,
                 features_metadata: dict,
                 feature_transformer_dim: int = TabNetConfig.feature_transformer_dim,
                 decision_step_output_dim: int = TabNetConfig.decision_step_output_dim,
                 num_decision_steps: int = TabNetConfig.num_decision_steps,
                 num_shared_layers: int = TabNetConfig.num_shared_layers,
                 num_decision_dependent_layers: int = TabNetConfig.num_decision_dependent_layers,
                 relaxation_factor: float = TabNetConfig.relaxation_factor,
                 batch_momentum: float = TabNetConfig.batch_momentum,
                 virtual_batch_size: int = TabNetConfig.virtual_batch_size,
                 residual_normalization_factor: float = TabNetConfig.residual_normalization_factor,
                 epsilon: float = TabNetConfig.epsilon,
                 encode_categorical_values: bool = TabNetConfig.encode_categorical_features,
                 **kwargs):
        categorical_feature_embedding = CategoricalFeatureEmbedding(features_metadata=features_metadata,
                                                                     embedding_dim=1,
                                                                     encode=encode_categorical_values)
        encoder = TabNetEncoder(data_dim=input_dim,
                                feature_transformer_dim=feature_transformer_dim,
                                decision_step_output_dim=decision_step_output_dim,
                                num_decision_steps=num_decision_steps,
                                num_shared_layers=num_shared_layers,
                                num_decision_dependent_layers=num_decision_dependent_layers,
                                relaxation_factor=relaxation_factor,
                                batch_momentum=batch_momentum,
                                virtual_batch_size=virtual_batch_size,
                                residual_normalization_factor=residual_normalization_factor,
                                epsilon=epsilon)
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
                         categorical_feature_embedding=categorical_feature_embedding,
                         encoder=encoder,
                         **kwargs)
        self.input_dim = input_dim
        self.features_metadata = features_metadata
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_output_dim = decision_step_output_dim
        self.num_decision_steps = num_decision_steps
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.residual_normalization_factor = residual_normalization_factor
        self.epsilon = epsilon
        self.encode_categorical_values = encode_categorical_values

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'features_metadata': self.features_metadata,
                  'feature_transformer_dim': self.feature_transformer_dim,
                  'decision_step_output_dim': self.decision_step_output_dim,
                  'num_decision_steps': self.num_decision_steps,
                  'num_shared_layers': self.num_shared_layers,
                  'num_decision_dependent_layers': self.num_decision_dependent_layers,
                  'relaxation_factor': self.relaxation_factor,
                  'batch_momentum': self.batch_momentum,
                  'virtual_batch_size': self.virtual_batch_size,
                  'residual_normalization_factor': self.residual_normalization_factor,
                  'epsilon': self.epsilon,
                  'encode_categorical_values': self.encode_categorical_values,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        features_metadata = config.pop("features_metadata")
        return cls(input_dim=input_dim,
                   features_metadata=features_metadata,
                   **config)


@keras.saving.register_keras_serializable(package="keras.models")
class TabNetClassifier(TabNet):
    """
    TabNet Classifier based on the TabNet architecture
    as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        num_classes: ``int``, default 2,
            Number of classes to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the input dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            ``{feature_name: (feature_idx, vocabulary)}`` for feature in categorical features.
            ``{feature_name: feature_idx}`` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        feature_transformer_dim: ``int``, default 32,
            The dimensionality of the hidden representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.

        decision_step_output_dim: ``int``, default 32,
            The dimensionality of output at each decision step, which is later mapped to the
            final classification or regression output.
            It is recommended to keep ``decision_step_output_dim`` and ``feature_transformer_dim``
            equal to each other.
            Adjusting these two parameters values is a good way of obtaining a tradeoff between
            performance and complexity.

        num_decision_steps: ``int``, default 5,
            The number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset, the value tends to
            be higher. That said, a very high value of `num_decision_steps` may suffer
            from overfitting.

        num_shared_layers: ``int``, default 2,
            Number of shared layers to use in the ``TabNetFeatureTransformer``.
            These shared layers are *shared* across decision steps.

        num_decision_dependent_layers: ``int``, default 2,
            Number of decision dependent layers to use in the ``TabNetFeatureTransformer``.
            In simple words, ``num_decision_dependent_layers`` are created
            for each decision step in the ``num_decision_steps``.
            For instance, if ``num_decision_steps = `5` and  ``num_decision_dependent_layers = 2``
            then 10 layers will be created, 2 for each decision step.

        relaxation_factor: ``float``, default 1.5,
            Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger ``relaxation_factor``.

        batch_momentum: ``float``, default 0.9,
            Momentum value to use for ``BatchNormalization`` layer.

        virtual_batch_size: `int`, default 64,
            Batch size to use for ``virtual_batch_size`` parameter in ``BatchNormalization`` layer.
            This is typically much smaller than the ``batch_size`` used for training.

        residual_normalization_factor: ``float``, default 0.5,
            In the feature transformer, except for the first layer, every other layer utilizes
            normalized residuals, where ``residual_normalization_factor``
            determines the scale of normalization.

        epsilon: ``float``, default 0.00001,
            Epsilon is a small number for numerical stability
            during the computation of entropy loss.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 features_metadata: dict = None,
                 feature_transformer_dim: int = TabNetConfig.feature_transformer_dim,
                 decision_step_output_dim: int = TabNetConfig.decision_step_output_dim,
                 num_decision_steps: int = TabNetConfig.num_decision_steps,
                 num_shared_layers: int = TabNetConfig.num_shared_layers,
                 num_decision_dependent_layers: int = TabNetConfig.num_decision_dependent_layers,
                 relaxation_factor: float = TabNetConfig.relaxation_factor,
                 batch_momentum: float = TabNetConfig.batch_momentum,
                 virtual_batch_size: int = TabNetConfig.virtual_batch_size,
                 residual_normalization_factor: float = TabNetConfig.residual_normalization_factor,
                 epsilon: float = TabNetConfig.epsilon,
                 encode_categorical_values: bool = TabNetConfig.encode_categorical_features,
                 **kwargs):
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_values,
                                  name="tabnet_classification_head")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
                         feature_transformer_dim=feature_transformer_dim,
                         decision_step_output_dim=decision_step_output_dim,
                         num_decision_steps=num_decision_steps,
                         num_shared_layers=num_shared_layers,
                         num_decision_dependent_layers=num_decision_dependent_layers,
                         relaxation_factor=relaxation_factor,
                         batch_momentum=batch_momentum,
                         virtual_batch_size=virtual_batch_size,
                         residual_normalization_factor=residual_normalization_factor,
                         epsilon=epsilon,
                         encode_categorical_values=encode_categorical_values,
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


@keras.saving.register_keras_serializable(package="keras.models")
class TabNetRegressor(TabNet):
    """
    TabNet Regressor based on the TabNet architecture
    as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        num_outputs: ``int``, default 1,
            Number of regression outputs.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the input dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            ``{feature_name: (feature_idx, vocabulary)}`` for feature in categorical features.
            ``{feature_name: feature_idx}`` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        feature_transformer_dim: ``int``, default 32,
            The dimensionality of the hidden representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.

        decision_step_output_dim: ``int``, default 32,
            The dimensionality of output at each decision step, which is later mapped to the
            final classification or regression output.
            It is recommended to keep ``decision_step_output_dim`` and ``feature_transformer_dim``
            equal to each other.
            Adjusting these two parameters values is a good way of obtaining a tradeoff between
            performance and complexity.

        num_decision_steps: ``int``, default 5,
            The number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset, the value tends to
            be higher. That said, a very high value of `num_decision_steps` may suffer
            from overfitting.

        num_shared_layers: ``int``, default 2,
            Number of shared layers to use in the ``TabNetFeatureTransformer``.
            These shared layers are *shared* across decision steps.

        num_decision_dependent_layers: ``int``, default 2,
            Number of decision dependent layers to use in the ``TabNetFeatureTransformer``.
            In simple words, ``num_decision_dependent_layers`` are created
            for each decision step in the ``num_decision_steps``.
            For instance, if ``num_decision_steps = `5` and  ``num_decision_dependent_layers = 2``
            then 10 layers will be created, 2 for each decision step.

        relaxation_factor: ``float``, default 1.5,
            Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger ``relaxation_factor``.

        batch_momentum: ``float``, default 0.9,
            Momentum value to use for ``BatchNormalization`` layer.

        virtual_batch_size: `int`, default 64,
            Batch size to use for ``virtual_batch_size`` parameter in ``BatchNormalization`` layer.
            This is typically much smaller than the ``batch_size`` used for training.

        residual_normalization_factor: ``float``, default 0.5,
            In the feature transformer, except for the first layer, every other layer utilizes
            normalized residuals, where ``residual_normalization_factor``
            determines the scale of normalization.

        epsilon: ``float``, default 0.00001,
            Epsilon is a small number for numerical stability
            during the computation of entropy loss.

        encode_categorical_values: ``bool``, default True,
            Whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's ``IntegerLookup`` layer.
    """
    def __init__(self,
                 num_outputs=1,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 features_metadata: dict = None,
                 feature_transformer_dim: int = TabNetConfig.feature_transformer_dim,
                 decision_step_output_dim: int = TabNetConfig.decision_step_output_dim,
                 num_decision_steps: int = TabNetConfig.num_decision_steps,
                 num_shared_layers: int = TabNetConfig.num_shared_layers,
                 num_decision_dependent_layers: int = TabNetConfig.num_decision_dependent_layers,
                 relaxation_factor: float = TabNetConfig.relaxation_factor,
                 batch_momentum: float = TabNetConfig.batch_momentum,
                 virtual_batch_size: int = TabNetConfig.virtual_batch_size,
                 residual_normalization_factor: float = TabNetConfig.residual_normalization_factor,
                 epsilon: float = TabNetConfig.epsilon,
                 encode_categorical_values: bool = TabNetConfig.encode_categorical_features,
                 **kwargs):
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_values,
                              name="tabnet_regression_head")
        super().__init__(input_dim=input_dim,
                         features_metadata=features_metadata,
                         feature_transformer_dim=feature_transformer_dim,
                         decision_step_output_dim=decision_step_output_dim,
                         num_decision_steps=num_decision_steps,
                         num_shared_layers=num_shared_layers,
                         num_decision_dependent_layers=num_decision_dependent_layers,
                         relaxation_factor=relaxation_factor,
                         batch_momentum=batch_momentum,
                         virtual_batch_size=virtual_batch_size,
                         residual_normalization_factor=residual_normalization_factor,
                         epsilon=epsilon,
                         encode_categorical_values=encode_categorical_values,
                         head=head,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_outputs': self.num_outputs,
                       'head_units_values': self.head_units_values
                       })
        return config


@keras.saving.register_keras_serializable(package="keras.models")
class TabNetPretrainer(_TabNetPretrainerLF):
    """
    TabNetPretrainer model based on the architecture
    proposed by Sercan et al. in TabNet paper.

    TabNetPretrainer is an encoder-decoder model based on the TabNet architecture,
    where the TabNet model acts as an encoder while a separate decoder
    is used to reconstruct the input features.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        model: ``TabNet``,
            An instance of `TabNet` class to pretrain.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the input dataset.

        features_metadata: ``dict``,
            A nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            ``{feature_name: (feature_idx, vocabulary)}`` for feature in categorical features.
            ``{feature_name: feature_idx}`` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                ..                                                      categorical_features,
                ..                                                      numerical_features)

        missing_feature_probability: ``float``, default 3,
            Fraction of features to randomly mask i.e. make them missing.
            Missing features are introduced in the pretraining dataset and
            the probability of missing features is controlled by the parameter.
            The pretraining objective is to predict values for these missing features,
            (pre)training the ``TabNet`` model in the process.

        decoder_feature_transformer_dim: ``int``, default 32,
            Feature transformer dimensions for ``TabNetDecoder``.

        decoder_decision_step_output_dim: ``int``, default 32,
            Decision step output dimensions for ``TabNetDecoder``.

        decoder_num_decision_steps: ``int``, default 5,
            Number of decision steps to use in ``TabNetDecoder``.

        decoder_num_shared_layers: ``int``, default 2,
            Number of shared layers in ``TabNetFeatureTransformer`` in ``TabNetDecoder``.

        decoder_num_decision_dependent_layers: ``int``, default 2,
            Number of decision dependent layers in ``TabNetFeatureTransformer`` layer in ``TabNetDecoder``.
    """
    def __init__(self,
                 model: TabNet,
                 input_dim: int = None,
                 features_metadata: dict = None,
                 missing_feature_probability: float = 0.3,
                 decoder_feature_transformer_dim: int = TabNetConfig.feature_transformer_dim,
                 decoder_decision_step_output_dim: int = TabNetConfig.decision_step_output_dim,
                 decoder_num_decision_steps: int = TabNetConfig.num_decision_steps,
                 decoder_num_shared_layers: int = TabNetConfig.num_shared_layers,
                 decoder_num_decision_dependent_layers: int = TabNetConfig.num_decision_dependent_layers,
                 **kwargs):
        decoder = TabNetDecoder(data_dim=input_dim,
                                feature_transformer_dim=decoder_feature_transformer_dim,
                                decision_step_output_dim=decoder_decision_step_output_dim,
                                num_decision_steps=decoder_num_decision_steps,
                                num_shared_layers=decoder_num_shared_layers,
                                num_decision_dependent_layers=decoder_num_shared_layers)
        super().__init__(model=model,
                         features_metadata=features_metadata,
                         decoder=decoder,
                         missing_feature_probability=missing_feature_probability,
                         **kwargs)
        self.model = model
        self.input_dim = input_dim
        self.features_metadata = features_metadata
        self.missing_feature_probability = missing_feature_probability
        self.decoder_feature_transformer_dim = decoder_feature_transformer_dim
        self.decoder_decision_step_output_dim = decoder_decision_step_output_dim
        self.decoder_num_decision_steps = decoder_num_decision_steps
        self.decoder_num_shared_layers = decoder_num_shared_layers
        self.decoder_num_decision_dependent_layers = decoder_num_decision_dependent_layers

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'model': keras.layers.serialize(self.model),
                  'input_dim': self.input_dim,
                  'features_metadata': self.features_metadata,
                  'missing_feature_probability': self.missing_feature_probability,
                  'decoder_feature_transformer_dim': self.decoder_feature_transformer_dim,
                  'decoder_decision_step_output_dim': self.decoder_decision_step_output_dim,
                  'decoder_num_decision_steps': self.decoder_num_decision_steps,
                  'decoder_num_shared_layers': self.decoder_num_shared_layers,
                  'decoder_num_decision_dependent_layers': self.decoder_num_decision_dependent_layers,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        model = keras.layers.deserialize(config.pop("model"))
        return cls(model=model, **config)
