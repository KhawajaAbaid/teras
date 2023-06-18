import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers import TabNetEncoder
from teras.layers.tabnet import Decoder
from typing import Union, List
import tensorflow_probability as tfp
from teras.losses.tabnet import reconstruction_loss
from teras.layers import CategoricalFeaturesEmbedding
from teras.config.tabnet import TabNetConfig, TabNetClassifierConfig, TabNetRegressorConfig, TabNetPreTrainerConfig

LAYER_OR_MODEL = Union[keras.layers.Layer, keras.Model]
LIST_OF_STR = List[str]


class TabNet(keras.Model):
    """
    TabNet model as proposed by Sercan et al. in TabNet paper.
    This purpose will serve as the parent class for the TabNetRegressor and TabNetClassifier.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
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
    """
    def __init__(self,
                 categorical_features_vocabulary: dict = TabNetConfig.categorical_features_vocabulary,
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
                 **kwargs):
        super().__init__(**kwargs)
        self.categorical_features_vocabulary = categorical_features_vocabulary
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

        self.categorical_features_embedding = CategoricalFeaturesEmbedding(
                                                                self.categorical_features_vocabulary,
                                                                embedding_dim=1)

        self.encoder = TabNetEncoder(feature_transformer_dim=self.feature_transformer_dim,
                                     decision_step_output_dim=self.decision_step_output_dim,
                                     num_decision_steps=self.num_decision_steps,
                                     num_shared_layers=self.num_shared_layers,
                                     num_decision_dependent_layers=self.num_decision_dependent_layers,
                                     relaxation_factor=self.relaxation_factor,
                                     batch_momentum=self.batch_momentum,
                                     virtual_batch_size=self.virtual_batch_size,
                                     residual_normalization_factor=self.residual_normalization_factor,
                                     epsilon=self.epsilon)

        self.is_pretrained = False
        self.pretrainer = None

    def pretrain(self,
                 pretraining_dataset,
                 num_features: int = None,
                 missing_feature_probability: float = 0.3,
                 pretraining_epochs: int = 10):
        """

        Args:
            pretraining_dataset: Dataset used for pretraining. It doesn't have to be labeled.
            missing_feature_probability: Missing features are introduced in the pretraining
                dataset and the probability of missing features is controlled by the parameter.
                The pretraining objective is to predict values for these missing features,
                (pre)training the encoder in process.
            pretraining_epochs: `int`, default 10, Number of epochs to pretrain the model for.
            fit_kwargs: Any other keyword arguments taken by the `fit` method of a Keras model.
        """
        dim = num_features
        # dim = tf.shape(pretraining_dataset)[1]
        pretrainer = TabNetPretrainer(data_dim=dim,
                                           missing_feature_probability=missing_feature_probability,
                                           categorical_features_vocabulary=self.categorical_features_vocabulary,
                                           encoder_feature_transformer_dim=self.feature_transformer_dim,
                                           encoder_decision_step_output_dim=self.decision_step_output_dim,
                                           encoder_num_decision_steps=self.num_decision_steps,
                                           encoder_num_shared_layers=self.num_shared_layers,
                                           encoder_num_decision_dependent_layers=self.num_decision_dependent_layers,

                                           decoder_feature_transformer_dim=self.feature_transformer_dim,
                                           decoder_decision_step_output_dim=self.decision_step_output_dim,
                                           decoder_num_decision_steps=self.num_decision_steps,
                                           decoder_num_shared_layers=self.num_shared_layers,
                                           decoder_num_decision_dependent_layers= self.num_decision_dependent_layers,

                                           virtual_batch_size=self.virtual_batch_size,
                                           batch_momentum=self.batch_momentum,
                                           residual_normalization_factor=self.residual_normalization_factor)
        pretrainer.compile()
        pretrainer.fit(pretraining_dataset, epochs=pretraining_epochs)
        self.is_pretrained = True
        self.encoder = pretrainer.get_encoder()

    def call(self, inputs):
        embedded_inputs = self.categorical_features_embedding(inputs)
        outputs = self.encoder(embedded_inputs)
        return outputs


class TabNetClassifier(TabNet):
    """
    TabNet Classifier based on the TabNet architecture as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        num_classes: Number of classes to predict.
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
    """
    def __init__(self,
                 num_classes=2,
                 feature_transformer_dim: int = 32,
                 decision_step_output_dim: int = 32,
                 num_decision_steps: int = 5,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 relaxation_factor: float = 1.5,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 epsilon=1e-5,
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
                         **kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes

        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        self.output_layer = layers.Dense(self.num_classes, activation=activation)

    def call(self, inputs):
        outputs = self.encoder(inputs)
        predictions = self.output_layer(outputs)
        return predictions


class TabNetRegressor(TabNet):
    """
    TabNet Regressor based on the TabNet architecture as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        num_outputs: Number of regression outputs.
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
    """
    def __init__(self,
                 num_outputs=1,
                 categorical_features_vocabulary: dict = None,
                 feature_transformer_dim: int = 32,
                 decision_step_output_dim: int = 32,
                 num_decision_steps: int = 5,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 relaxation_factor: float = 1.5,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(categorical_features_vocabulary,
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
                         **kwargs)
        self.num_outputs = num_outputs
        self.head = layers.Dense(self.num_outputs, name="tabnet_regressor_head")

    def call(self, inputs):
        encoded_features = super().__call__(inputs)
        predictions = self.head(encoded_features)
        return predictions


class TabNetPretrainer(keras.Model):
    """
    TabNet model as proposed by Sercan et al. in TabNet paper.
    This purpose will serve as the parent class for the TabNetRegressor and TabNetClassifier.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: `int`, Dimensionality of the input dataset.
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
    """

    def __init__(self,
                 data_dim: int,
                 missing_feature_probability: float = 0.3,
                 categorical_features_vocabulary: dict = None,

                 encoder_feature_transformer_dim: int = 32,
                 encoder_decision_step_output_dim: int = 32,
                 encoder_num_decision_steps: int = 5,
                 encoder_num_shared_layers: int = 2,
                 encoder_num_decision_dependent_layers: int = 2,

                 decoder_feature_transformer_dim: int = 32,
                 decoder_decision_step_output_dim: int = 32,
                 decoder_num_decision_steps: int = 5,
                 decoder_num_shared_layers: int = 2,
                 decoder_num_decision_dependent_layers: int = 2,

                 relaxation_factor: float = 1.5,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.missing_feature_probability = missing_feature_probability
        self.categorical_features_vocabulary = categorical_features_vocabulary

        self.encoder_feature_transformer_dim = encoder_feature_transformer_dim
        self.encoder_decision_step_output_dim = encoder_decision_step_output_dim
        self.encoder_num_decision_steps = encoder_num_decision_steps
        self.encoder_num_shared_layers = encoder_num_shared_layers
        self.encoder_num_decision_dependent_layers = encoder_num_decision_dependent_layers

        self.decoder_feature_transformer_dim = decoder_feature_transformer_dim
        self.decoder_decision_step_output_dim = decoder_decision_step_output_dim
        self.decoder_num_decision_steps = decoder_num_decision_steps
        self.decoder_num_shared_layers = decoder_num_shared_layers
        self.decoder_num_decision_dependent_layers = decoder_num_decision_dependent_layers

        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.residual_normalization_factor = residual_normalization_factor
        self.epsilon = epsilon

        self.binary_mask_generator = tfp.distributions.Binomial(total_count=1,
                                                                probs=self.missing_feature_probability,
                                                                name="binary_mask_generator")

        self.categorical_features_embedding = CategoricalFeaturesEmbedding(categorical_features_vocabulary,
                                                                           embedding_dim=1)

        self.encoder = TabNetEncoder(feature_transformer_dim=self.encoder_feature_transformer_dim,
                                     decision_step_output_dim=self.encoder_decision_step_output_dim,
                                     num_decision_steps=self.encoder_num_decision_steps,
                                     num_shared_layers=self.encoder_num_shared_layers,
                                     num_decision_dependent_layers=self.encoder_num_decision_dependent_layers,
                                     relaxation_factor=self.relaxation_factor,
                                     batch_momentum=self.batch_momentum,
                                     virtual_batch_size=self.virtual_batch_size,
                                     residual_normalization_factor=self.residual_normalization_factor,
                                     epsilon=self.epsilon)

        self.decoder = Decoder(data_dim=self.data_dim,
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
