import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from teras.layers import TabNetEncoder
from teras.layers.tabnet import Decoder
from typing import Union
import tensorflow_probability as tfp
from teras.losses.tabnet import reconstruction_loss


LAYER_OR_MODEL = Union[keras.layers.Layer, keras.Model]


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
        super().__init__(**kwargs)
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

    def call(self, inputs):
        outputs = self.encoder(inputs)
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
        self.num_outputs = num_outputs
        self.output_layer = layers.Dense(self.num_outputs)

    def call(self, inputs):
        encoded_features = self.encoder(inputs)
        predictions = self.output_layer(encoded_features)
        return predictions


class TabNetPretrainer(keras.Model):
    """
    TabNet model as proposed by Sercan et al. in TabNet paper.
    This purpose will serve as the parent class for the TabNetRegressor and TabNetClassifier.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: `int`, Dimensionality of the input dataset.
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
                 data_dim: int,
                 miss_probability: float = 0.3,
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
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.miss_probability = miss_probability
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

        self.binary_mask_generator = tfp.distributions.Binomial(total_count=1,
                                                                probability=self.miss_probability,
                                                                name="binary_mask_generator")

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

        self.decoder = Decoder(data_dim=self.data_dim,
                               feature_transformer_dim=self.feature_transformer_dim,
                               decision_step_output_dim=self.decision_step_output_dim,
                               num_decision_steps=self.num_decision_steps,
                               num_shared_layers=self.num_shared_layers,
                               num_decision_dependent_layers=self.num_decision_dependent_layers,
                               batch_momentum=self.batch_momentum,
                               virtual_batch_size=self.virtual_batch_size,
                               residual_normalization_factor=self.residual_normalization_factor,
                               )

    def compile(self,
                reconstruction_loss=reconstruction_loss,
                **kwargs):
        super().compile(**kwargs)
        self.reconstruction_loss = reconstruction_loss

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # this mask below is what `S` means in the paper, where if an index contains
        # value 1, it means that it is missing
        mask = self.binary_mask_generator(shape=(batch_size, self.data_dim))
        encoder_input = (1 - mask) * inputs
        # The paper says,
        # The TabNet encoder inputs (1 − S) · f
        # and the decoder outputs the reconstructed features, S · ^f
        # We initialize P[0] = (1 − S) in encoder so that the model emphasizes merely on the known features.
        # -- So we pass the mask from here, the encoder checks if it received a value for mask, if so it won't
        # initialized the `mask_values` variable in its call method to zeros.
        encoded_representations = self.encoder(encoder_input, mask=(1 - mask))
        decoder_outputs = self.decoder(encoded_representations)
        loss = self.reconstruction_loss(real_samples=inputs,
                                        reconstructed_samples=decoder_outputs,
                                        mask=mask)
        self.add_loss(loss)

