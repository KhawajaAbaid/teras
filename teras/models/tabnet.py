import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers import TabNetEncoder, TabNetDecoder
from typing import Union, List
import tensorflow_probability as tfp
from teras.losses.tabnet import reconstruction_loss
from teras.layers import CategoricalFeatureEmbedding
from teras.config.tabnet import TabNetConfig
from teras.config.base import FitConfig
from warnings import warn


LAYER_OR_MODEL = Union[keras.layers.Layer, keras.Model]
LIST_OF_STR = List[str]


class TabNet(keras.Model):
    """
    TabNet model as proposed by Sercan et al. in TabNet paper.
    This purpose will serve as the parent class for the TabNetRegressor and TabNetClassifier.

    Reference(s):
        https://arxiv.org/abs/1908.07442

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
        feature_transformer_dim: `int`, default 32, the dimensionality of the hidden
            representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.
        decision_step_output_dim: `int`, default 32, the dimensionality of output at each
            decision step, which is later mapped to the final classification or regression output.
            It is recommended to keep `decision_step_output_dim` and `feature_transformer_dim`
            equal to each other.
            Adjusting these two parameters values is a good way of obtaining a tradeoff between
            performance and complexity.
        num_decision_steps: `int`, default 5, the number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset, the value tends to
            be higher. That said, a very high value of `num_decision_steps` may suffer
            from overfitting.
        num_shared_layers: `int`, default 2. Number of shared layers to use in the Feature Transformer.
            These shared layers are `shared` across decision steps.
        num_decision_dependent_layers: `int`, default 2. Number of decision dependent layers to use in
            the Feature Transformer. In simple words, `num_decision_dependent_layers` are created
            for each decision step in the `num_decision_steps`.
            For instance, if `num_decision_steps = 5` and  `num_decision_dependent_layers = 2`
            then 10 layers will be created, 2 for each decision step.
        relaxation_factor: `float`, default 1.5, Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger `relaxation_factor`.
        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
            This is typically much smaller than the `batch_size` used for training.
        residual_normalization_factor: `float`, default 0.5, In the feature transformer, except for the
            layer, every other layer utilizes normalized residuals, where `residual_normalization_factor`
            determines the scale of normalization.
        epsilon: `float`, default 0.00001, Epsilon is a small number for numerical stability
            during the computation of entropy loss.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
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
        super().__init__(**kwargs)

        if features_metadata is None:
            raise ValueError(f"""
            `features_metadata` is required for embedding features and hence cannot be None.            
            You can get this features_metadata dictionary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features, numerical_features)`
            Received, `features_metadata`: {features_metadata}
                 """)

        if not encode_categorical_values:
            warn("`encode_categorical_values` is set to False. Categorical values are assumed to be encoded "
                 "and hence no encoding will be applied before embedding generation.")

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

        self._categorical_features_metadata = features_metadata["categorical"]
        self._numerical_features_metadata = features_metadata["numerical"]
        self._num_numerical_features = len(self._numerical_features_metadata)
        self._num_categorical_features = len(self._categorical_features_metadata)
        self._num_features = self._num_numerical_features + self._num_categorical_features

        self._numerical_features_exist = self._num_numerical_features > 0
        self._categorical_features_exist = self._num_categorical_features > 0

        self.categorical_features_embedding = CategoricalFeatureEmbedding(
            categorical_features_metadata=self._categorical_features_metadata,
            embedding_dim=1,
            encode=self.encode_categorical_values)

        self.encoder = TabNetEncoder(feature_transformer_dim=self.feature_transformer_dim,
                                     decision_step_output_dim=self.decision_step_output_dim,
                                     num_decision_steps=self.num_decision_steps,
                                     num_shared_layers=self.num_shared_layers,
                                     num_decision_dependent_layers=self.num_decision_dependent_layers,
                                     relaxation_factor=self.relaxation_factor,
                                     batch_momentum=self.batch_momentum,
                                     virtual_batch_size=self.virtual_batch_size,
                                     residual_normalization_factor=self.residual_normalization_factor,
                                     epsilon=self.epsilon,
                                     num_features=self._num_features)
        # PLEASE WORK!!
        # self.encoder.build(input_shape=(None, self._num_features))

        self.pretrainer_fit_config = FitConfig()

        self._pretrained = False
        self.pretrainer = None
        self.head = None

        self._categorical_features_names = None
        self._categorical_features_idx = None
        if self._categorical_features_exist:
            self._categorical_features_idx = set(map(lambda x: x[0], self._categorical_features_metadata.values()))
            self._categorical_features_names = set(self._categorical_features_metadata.keys())

        self._numerical_features_names = None
        self._numerical_features_idx = None

        self._is_first_batch = True
        self._is_data_in_dict_format = False

    def pretrain(self,
                 pretraining_dataset,
                 missing_feature_probability: float = 0.3,
                 decoder_feature_transformer_dim: int = 32,
                 decoder_decision_step_output_dim: int = 32,
                 decoder_num_decision_steps: int = 5,
                 decoder_num_shared_layers: int = 2,
                 decoder_num_decision_dependent_layers: int = 2,
                 ):
        """
        Helper function to pretrain the encoder and
        Args:
            pretraining_dataset: Dataset used for pretraining. It doesn't have to be labeled.
            missing_feature_probability: Missing features are introduced in the pretraining
                dataset and the probability of missing features is controlled by the parameter.
                The pretraining objective is to predict values for these missing features,
                (pre)training the encoder in process.
            decoder_feature_transformer_dim: `int`, default 32, Feature transformer dimensions for decoder.
            decoder_decision_step_output_dim: `int`, default 32, Decision step output dimensions for decoder.
            decoder_num_decision_steps: `int`, default 5, Number of decision steps to use in decoder.
            decoder_num_shared_layers: `int`, default 2, Number of shared layers in feature transformer in decoder.
            decoder_num_decision_dependent_layers: `int`, default 2, Number of decision dependent layers
                in feature transformer in decoder.
        """
        dim = self._num_features
        pretrainer = TabNetPretrainer(data_dim=dim,
                                      missing_feature_probability=missing_feature_probability,
                                      features_metadata=self.features_metadata,
                                      encoder_feature_transformer_dim=self.feature_transformer_dim,
                                      encoder_decision_step_output_dim=self.decision_step_output_dim,
                                      encoder_num_decision_steps=self.num_decision_steps,
                                      encoder_num_shared_layers=self.num_shared_layers,
                                      encoder_num_decision_dependent_layers=self.num_decision_dependent_layers,
                                      decoder_feature_transformer_dim=decoder_feature_transformer_dim,
                                      decoder_decision_step_output_dim=decoder_decision_step_output_dim,
                                      decoder_num_decision_steps=decoder_num_decision_steps,
                                      decoder_num_shared_layers=decoder_num_shared_layers,
                                      decoder_num_decision_dependent_layers=decoder_num_decision_dependent_layers,
                                      virtual_batch_size=self.virtual_batch_size,
                                      batch_momentum=self.batch_momentum,
                                      residual_normalization_factor=self.residual_normalization_factor,
                                      encode_categorical_values=self.encode_categorical_values,
                                    )
        pretrainer.compile()
        # print("passing params", self.pretrainer_fit_config.to_dict())
        pretrainer.fit(pretraining_dataset, **self.pretrainer_fit_config.to_dict())
        self.categorical_features_embedding = pretrainer.categorical_features_embedding
        self.encoder = pretrainer.get_encoder()
        self._pretrained = True

    def call(self, inputs):
        if self._is_first_batch:
            if isinstance(inputs, dict):
                self._is_data_in_dict_format = True
            self._is_first_batch = False

        features = None
        if self._categorical_features_exist:
            categorical_features = self.categorical_features_embedding(inputs)
            categorical_features = tf.squeeze(categorical_features)
            features = categorical_features

        if self._numerical_features_exist:
            # In TabNet we pass raw numerical feature to the encoder -- no embeddings are applied.
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

        features.set_shape([None, self._num_features])
        outputs = self.encoder(features)
        if self.head is not None:
            outputs = self.head(outputs)
        return outputs


class TabNetClassifier(TabNet):
    """
    TabNet Classifier based on the TabNet architecture as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        activation_out: Activation to use in the Classification head,
            by default, `sigmoid` is used for binary and `softmax` is used
            for multi-class classification.
        feature_transformer_dim: `int`, default 32, the dimensionality of the hidden
            representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.
        decision_step_output_dim: `int`, default 32, the dimensionality of output at each
            decision step, which is later mapped to the final classification or regression output.
            It is recommended to keep `decision_step_output_dim` and `feature_transformer_dim`
            equal to each other.
            Adjusting these two parameters values is a good way of obtaining a tradeoff between
            performance and complexity.
        num_decision_steps: `int`, default 5, the number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset, the value tends to
            be higher. That said, a very high value of `num_decision_steps` may suffer
            from overfitting.
        num_shared_layers: `int`, default 2. Number of shared layers to use in the Feature Transformer.
            These shared layers are `shared` across decision steps.
        num_decision_dependent_layers: `int`, default 2. Number of decision dependent layers to use in
            the Feature Transformer. In simple words, `num_decision_dependent_layers` are created
            for each decision step in the `num_decision_steps`.
            For instance, if `num_decision_steps = 5` and  `num_decision_dependent_layers = 2`
            then 10 layers will be created, 2 for each decision step.
        relaxation_factor: `float`, default 1.5, Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger `relaxation_factor`.
        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
            This is typically much smaller than the `batch_size` used for training.
        residual_normalization_factor: `float`, default 0.5, In the feature transformer, except for the
            layer, every other layer utilizes normalized residuals, where `residual_normalization_factor`
            determines the scale of normalization.
        epsilon: `float`, default 0.00001, Epsilon is a small number for numerical stability
            during the computation of entropy loss.
        categorical_features_vocabulary: `dict`, Vocabulary of categorical feature.
            Vocabulary is simply a dictionary where feature name maps
            to a tuple of feature index and a list of unique values in the feature.
            You can get this vocabulary by calling
            `teras.utils.get_categorical_features_vocabulary(dataset, categorical_features)`
            If None, dataset will be assumed to contain no categorical features and
            hence CategoricalFeatureEmbedding layer won't be applied.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
                 num_classes: int = 2,
                 activation_out=None,
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
                 categorical_features_vocabulary: dict = TabNetConfig.encode_categorical_features,
                 encode_categorical_values: bool = TabNetConfig.encode_categorical_features,
                 **kwargs):
        super().__init__(feature_transformer_dim=feature_transformer_dim,
                         decision_step_output_dim=decision_step_output_dim,
                         num_decision_steps=num_decision_steps,
                         num_shared_layers=num_shared_layers,
                         num_decision_dependent_layers=num_decision_dependent_layers,
                         relaxation_factor=relaxation_factor,
                         batch_momentum=batch_momentum,
                         virtual_batch_size=virtual_batch_size,
                         residual_normalization_factor=residual_normalization_factor,
                         epsilon=epsilon,
                         categorical_features_vocabulary=categorical_features_vocabulary,
                         encode_categorical_values=encode_categorical_values,
                         **kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = "sigmoid" if self.num_classes == 1 else "softmax"
        self.head = layers.Dense(self.num_classes, activation=self.activation_out)

    # def call(self, inputs):
    #     x = inputs
    #     if self.categorical_features_vocabulary is not None:
    #         x = self.categorical_features_embedding(x)
    #     x = self.encoder(x)
    #     predictions = self.output_layer(x)
    #     return predictions


class TabNetRegressor(TabNet):
    """
    TabNet Regressor based on the TabNet architecture as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        num_outputs: Number of regression outputs.
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
        feature_transformer_dim: `int`, default 32, the dimensionality of the hidden
            representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.
        decision_step_output_dim: `int`, default 32, the dimensionality of output at each
            decision step, which is later mapped to the final classification or regression output.
            It is recommended to keep `decision_step_output_dim` and `feature_transformer_dim`
            equal to each other.
            Adjusting these two parameters values is a good way of obtaining a tradeoff between
            performance and complexity.
        num_decision_steps: `int`, default 5, the number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset, the value tends to
            be higher. That said, a very high value of `num_decision_steps` may suffer
            from overfitting.
        num_shared_layers: `int`, default 2. Number of shared layers to use in the Feature Transformer.
            These shared layers are `shared` across decision steps.
        num_decision_dependent_layers: `int`, default 2. Number of decision dependent layers to use in
            the Feature Transformer. In simple words, `num_decision_dependent_layers` are created
            for each decision step in the `num_decision_steps`.
            For instance, if `num_decision_steps = 5` and  `num_decision_dependent_layers = 2`
            then 10 layers will be created, 2 for each decision step.
        relaxation_factor: `float`, default 1.5, Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger `relaxation_factor`.
        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
            This is typically much smaller than the `batch_size` used for training.
        residual_normalization_factor: `float`, default 0.5, In the feature transformer, except for the
            layer, every other layer utilizes normalized residuals, where `residual_normalization_factor`
            determines the scale of normalization.
        epsilon: `float`, default 0.00001, Epsilon is a small number for numerical stability
            during the computation of entropy loss.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """
    def __init__(self,
                 num_outputs=1,
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
        super().__init__(features_metadata=features_metadata,
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
                         **kwargs)
        self.num_outputs = num_outputs
        self.head = layers.Dense(self.num_outputs, name="tabnet_regressor_head")

    # def call(self, inputs):
    #     x = inputs
    #     if self.categorical_features_vocabulary is not None:
    #         x = self.categorical_features_embedding(x)
    #     x = self.encoder(x)
    #     predictions = self.head(x)
    #     return predictions


class TabNetPretrainer(TabNet):
    """
    TabNetPretrainer model based on the architecture
    proposed by Sercan et al. in TabNet paper.

    TabNetPretrainer subclasses the TabNet class since TabNet itself is just an encoder
    model while the TabNet decoder is an encoder-decoder model, so instead of instantiating
    everything encoder part specific here, we can just utilize the parent class i.e. TabNet
    which already implements it all and will serve as a useful abstraction to keep things clean.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: `int`, Dimensionality of the input dataset.
        missing_feature_probability: `float`, default 3, Fraction of features to randomly mask
            -- i.e. make them missing.
            Missing features are introduced in the pretraining dataset and
            the probability of missing features is controlled by the parameter.
            The pretraining objective is to predict values for these missing features,
            (pre)training the TabNet model in the process.
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
        encoder_feature_transformer_dim: `int`, default 32, the dimensionality of the hidden
            representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.
        encoder_decision_step_output_dim: `int`, default 32, the dimensionality of output at each
            decision step, which is later mapped to the final classification or regression output.
            It is recommended to keep `decision_step_output_dim` and `feature_transformer_dim`
            equal to each other.
            Adjusting these two parameters values is a good way of obtaining a tradeoff between
            performance and complexity.
        encoder_num_decision_steps: `int`, default 5, the number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset, the value tends to
            be higher. That said, a very high value of `num_decision_steps` may suffer
            from overfitting.
        encoder_num_shared_layers: `int`, default 2. Number of shared layers to use in the Feature Transformer.
            These shared layers are `shared` across decision steps.
        encoder_num_decision_dependent_layers: `int`, default 2. Number of decision dependent layers to use in
            the Feature Transformer. In simple words, `num_decision_dependent_layers` are created
            for each decision step in the `num_decision_steps`.
            For instance, if `num_decision_steps = 5` and  `num_decision_dependent_layers = 2`
            then 10 layers will be created, 2 for each decision step.

        decoder_feature_transformer_dim: Feature transformer dimensions for decoder.
        decoder_decision_step_output_dim: Decision step output dimensions for decoder.
        decoder_num_decision_steps: Number of decision steps to use in decoder.
        decoder_num_shared_layers: Number of shared layers in feature transformer in decoder.
        decoder_num_decision_dependent_layers: Number of decision dependent layers in feature transformer in decoder.

        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
            This is typically much smaller than the `batch_size` used for training.
        residual_normalization_factor: `float`, default 0.5, In the feature transformer, except for the
            layer, every other layer utilizes normalized residuals, where `residual_normalization_factor`
            determines the scale of normalization.
        relaxation_factor: `float`, default 1.5, Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger `relaxation_factor`.
        epsilon: `float`, default 0.00001, Epsilon is a small number for numerical stability
            during the computation of entropy loss.
        encode_categorical_values: `bool`, default True, whether to (label) encode categorical values,
            If you've already encoded the categorical values using for instance
            Label/Ordinal encoding, you should set this to False,
            otherwise leave it as True.
            In the case of True, categorical values will be mapped to integer indices
            using keras's string lookup layer.
    """

    def __init__(self,
                 data_dim: int,
                 missing_feature_probability: float = 0.3,
                 features_metadata: dict = None,

                 encoder_feature_transformer_dim: int = TabNetConfig.feature_transformer_dim,
                 encoder_decision_step_output_dim: int = TabNetConfig.decision_step_output_dim,
                 encoder_num_decision_steps: int = TabNetConfig.num_decision_steps,
                 encoder_num_shared_layers: int = TabNetConfig.num_shared_layers,
                 encoder_num_decision_dependent_layers: int = TabNetConfig.num_decision_dependent_layers,

                 decoder_feature_transformer_dim: int = TabNetConfig.feature_transformer_dim,
                 decoder_decision_step_output_dim: int = TabNetConfig.decision_step_output_dim,
                 decoder_num_decision_steps: int = TabNetConfig.num_decision_steps,
                 decoder_num_shared_layers: int = TabNetConfig.num_shared_layers,
                 decoder_num_decision_dependent_layers: int = TabNetConfig.num_decision_dependent_layers,

                 relaxation_factor: float = TabNetConfig.relaxation_factor,
                 batch_momentum: float = TabNetConfig.batch_momentum,
                 virtual_batch_size: int = TabNetConfig.virtual_batch_size,
                 residual_normalization_factor: float = TabNetConfig.residual_normalization_factor,
                 epsilon: float = TabNetConfig.epsilon,

                 encode_categorical_values: bool = TabNetConfig.encode_categorical_features,
                 **kwargs):
        # Since the base TabNet model is basically an Encoder only model
        # so instead of defining encoder specific things here,
        # we just subclass the TabNet class
        super().__init__(features_metadata=features_metadata,
                         feature_transformer_dim=encoder_feature_transformer_dim,
                         decision_step_output_dim=encoder_decision_step_output_dim,
                         num_decision_steps=encoder_num_decision_steps,
                         num_shared_layers=encoder_num_shared_layers,
                         num_decision_dependent_layers=encoder_num_decision_dependent_layers,
                         relaxation_factor=relaxation_factor,
                         batch_momentum=batch_momentum,
                         virtual_batch_size=virtual_batch_size,
                         residual_normalization_factor=residual_normalization_factor,
                         epsilon=epsilon,
                         encode_categorical_values=encode_categorical_values,
                         **kwargs)

        self.data_dim = data_dim
        self.missing_feature_probability = missing_feature_probability

        self.decoder_feature_transformer_dim = decoder_feature_transformer_dim
        self.decoder_decision_step_output_dim = decoder_decision_step_output_dim
        self.decoder_num_decision_steps = decoder_num_decision_steps
        self.decoder_num_shared_layers = decoder_num_shared_layers
        self.decoder_num_decision_dependent_layers = decoder_num_decision_dependent_layers

        self.binary_mask_generator = tfp.distributions.Binomial(total_count=1,
                                                                probs=self.missing_feature_probability,
                                                                name="binary_mask_generator")

        self.decoder = TabNetDecoder(data_dim=self.data_dim,
                                     feature_transformer_dim=self.decoder_feature_transformer_dim,
                                     decision_step_output_dim=self.decoder_decision_step_output_dim,
                                     num_decision_steps=self.decoder_num_decision_steps,
                                     num_shared_layers=self.decoder_num_shared_layers,
                                     num_decision_dependent_layers=self.decoder_num_decision_dependent_layers,
                                     batch_momentum=self.batch_momentum,
                                     virtual_batch_size=self.virtual_batch_size,
                                     residual_normalization_factor=self.residual_normalization_factor,
                                     )

        self._reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @property
    def feature_importances_per_sample(self):
        """Returns feature importances per sample computed during training."""
        return tf.concat(self.encoder.feature_importances_per_sample, axis=0)

    @property
    def feature_importances(self):
        """Returns average feature importanes across samples computed during training."""
        return tf.reduce_mean(self.feature_importances_per_sample, axis=0)

    def compile(self,
                loss=reconstruction_loss,
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                **kwargs):
        super().compile(**kwargs)
        self.reconstruction_loss = loss
        self.optimizer = optimizer

    def call(self, inputs, mask=None):
        # this mask below is what `S` means in the paper, where if an index contains
        # value 1, it means that it is missing
        encoder_input = (1 - mask) * inputs
        # The paper says,
        # The TabNet encoder inputs (1 − S) · f, where f is the original features
        # and the decoder outputs the reconstructed features, S · f^, where f^ is the reconstructed features
        # We initialize P[0] = (1 − S) in encoder so that the model emphasizes merely on the known features.
        # -- So we pass the mask from here, the encoder checks if it received a value for mask, if so it won't
        # initialized the `mask_values` variable in its call method to zeros.
        encoded_representations = self.encoder(encoder_input, mask=(1 - mask))
        decoder_outputs = self.decoder(encoded_representations)
        return decoder_outputs

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            embedded_inputs = self.categorical_features_embedding(data)
            batch_size = tf.shape(embedded_inputs)[0]
            # Generate mask to create missing samples
            mask = self.binary_mask_generator.sample(sample_shape=(batch_size, self.data_dim))
            tape.watch(mask)
            # Reconstruct samples
            reconstructed_samples = self(embedded_inputs, mask=mask)
            # Compute reconstruction loss
            loss = self.reconstruction_loss(real_samples=embedded_inputs,
                                            reconstructed_samples=reconstructed_samples,
                                            mask=mask)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self._reconstruction_loss_tracker.update_state(loss)
        # If user has passed any additional metrics to compile, we should update their states
        if len(self.compiled_metrics.metrics) > 0:
            self.compiled_metrics.update_state(embedded_inputs, reconstructed_samples)
        # If user has passed any additional losses to compile, we should call them
        if self.compiled_loss._losses is not None:
            self.compiled_loss(embedded_inputs, reconstructed_samples)
        results = {m.name: m.result() for m in self.metrics}
        return results
