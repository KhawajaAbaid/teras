from tensorflow import keras
from tensorflow.keras import layers
from teras.layers import TabNetEncoder
from typing import Union


LAYER_OR_MODEL = Union[keras.layers.Layer, keras.Model]


class TabNet(keras.Model):
    """
    TabNet model as proposed by Sercan et al. in TabNet paper.
    This purpose will serve as the parent class for the TabNetRegressor and TabNetClassifier.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        feature_transformer_dim: Number of hidden units to use in Fully Connected (Dense) layer of Feature Transformer
        decision_step_output_dim: Output dimensionality for the decision step
        num_decision_steps: Number of decision steps.
                            According to the paper, "TabNet uses sequential attention to choose which features
                            to reason from at each decision step"
        relaxation_factor: When = 1, a feature is enforced to be used only at one decision step
                        and as it increases, more flexibility is provided to use a feature at multiple decision steps.
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
        epsilon: Epsilon is a small number for numerical stability.
    """
    def __init__(self,
                 feature_transformer_dim=64,
                 decision_step_output_dim: int = 64,
                 num_decision_steps: int = 5,
                 relaxation_factor=1.5,
                 batch_momentum=0.7,
                 virtual_batch_size: int = 128,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_output_dim = decision_step_output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

        self.encoder = TabNetEncoder(units=self.feature_transformer_dim,
                                     decision_step_output_dim=self.decision_step_output_dim,
                                     num_decision_steps=self.num_decision_steps,
                                     relaxation_factor=self.relaxation_factor,
                                     batch_momentum=self.batch_momentum,
                                     virtual_batch_size=self.virtual_batch_size,
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
        num_classes: Number of classes to predict
        feature_transformer_dim: Number of hidden units to use in Fully Connected (Dense) layer
        decision_step_output_dim: Output dimensionality for the decision step
        num_decision_steps: Number of decision steps.
                            According to the paper, "TabNet uses sequential attention to choose which features
                            to reason from at each decision step"
        relaxation_factor: When = 1, a feature is enforced to be used only at one decision step
                        and as it increases, more flexibility is provided to use a feature at multiple decision steps.
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
        epsilon: Epsilon is a small number for numerical stability.
        output_activation: Activation layer to use for classification output.
                        By default, Sigmoid is used for binary while Softmax is used for Multiclass classification.
    """
    def __init__(self,
                 num_classes=2,
                 feature_transformer_dim=64,
                 decision_step_output_dim: int = 64,
                 num_decision_steps: int = 5,
                 relaxation_factor=1.5,
                 batch_momentum=0.7,
                 virtual_batch_size: int = 128,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(feature_transformer_dim=feature_transformer_dim,
                         decision_step_output_dim=decision_step_output_dim,
                         num_decision_steps=num_decision_steps,
                         relaxation_factor=relaxation_factor,
                         batch_momentum=batch_momentum,
                         virtual_batch_size=virtual_batch_size,
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
        feature_transformer_dim: Number of hidden units to use in Fully Connected (Dense) layer in FeatureTransformer
        decision_step_output_dim: Output dimensionality for the decision step
        num_decision_steps: Number of decision steps.
                            According to the paper, "TabNet uses sequential attention to choose which features
                            to reason from at each decision step"
        relaxation_factor: When = 1, a feature is enforced to be used only at one decision step
                        and as it increases, more flexibility is provided to use a feature at multiple decision steps.
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
        epsilon: Epsilon is a small number for numerical stability.
    """
    def __init__(self,
                 num_outputs=1,
                 feature_transformer_dim=64,
                 decision_step_output_dim: int = 64,
                 num_decision_steps: int = 5,
                 relaxation_factor=1.5,
                 batch_momentum=0.7,
                 virtual_batch_size: int = 128,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(feature_transformer_dim=feature_transformer_dim,
                         decision_step_output_dim=decision_step_output_dim,
                         num_decision_steps=num_decision_steps,
                         relaxation_factor=relaxation_factor,
                         batch_momentum=batch_momentum,
                         virtual_batch_size=virtual_batch_size,
                         epsilon=epsilon,
                         **kwargs)
        self.num_outputs = num_outputs
        self.output_layer = layers.Dense(self.num_outputs)

    def call(self, inputs):
        encoded_features = self.encoder(inputs)
        predictions = self.output_layer(encoded_features)
        return predictions
