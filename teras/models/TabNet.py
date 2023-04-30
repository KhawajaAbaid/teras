from tensorflow import keras
from tensorflow.keras import layers
from teras.layers import TabNetEncoder



class TabNetClassifier(keras.Model):
    """
    TabNet Classifier based on the TabNet architecture as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        num_classes: Number of classes to predict
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
                 units,
                 num_classes,
                 decision_step_output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 epsilon,
                 output_activation=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_classes = num_classes
        self.decision_step_output_dim = decision_step_output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.output_activation = output_activation

        self.encoder = TabNetEncoder(self.units,
                                     self.decision_step_output_dim,
                                     self.num_decision_steps,
                                     self.relaxation_factor,
                                     self.batch_momentum,
                                     self.virtual_batch_size,
                                     self.epsilon)

        if self.output_activation is None:
            if num_classes <= 2:
                self.num_classes = 1
                self.act_out = "sigmoid"
            else:
                self.act_out = "softmax"
        else:
            self.act_out = self.output_activation

        self.dense_out = layers.Dense(self.num_classes, activation=self.act_out)

    def call(self, inputs, training=None, mask=None):
        inputs = inputs
        encoded_features = self.encoder(inputs,
                                        training=training)
        predictions = self.dense_out(encoded_features)
        print("Predictions: ", predictions)
        return predictions


class TabNetRegressor(keras.Model):
    """
    TabNet Regressor based on the TabNet architecture as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        decision_step_output_dim: Output dimensionality for the decision step
        num_decision_steps: Number of decision steps.
                            According to the paper, "TabNet uses sequential attention to choose which features
                            to reason from at each decision step"
        relaxation_factor: When = 1, a feature is enforced to be used only at one decision step
                        and as it increases, more flexibility is provided to use a feature at multiple decision steps.
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
        epsilon: Epsilon is a small number for numerical stability.
        num_outputs: Number of regression outputs.
    """
    def __init__(self,
                 units,
                 decision_step_output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 epsilon,
                 num_outputs=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.decision_step_output_dim = decision_step_output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_outputs = num_outputs
        self.epsilon = epsilon

        self.encoder = TabNetEncoder(self.units,
                                     self.decision_step_output_dim,
                                     self.num_decision_steps,
                                     self.relaxation_factor,
                                     self.batch_momentum,
                                     self.virtual_batch_size,
                                     self.epsilon)

        self.dense_out = layers.Dense(self.num_outputs)

    def call(self, inputs, training=None, mask=None):
        encoded_features = self.encoder(inputs,
                                        training=training)
        predictions = self.dense_out(encoded_features)
        return predictions