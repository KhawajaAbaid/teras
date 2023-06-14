import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from teras.layers import GLU
# from teras.activations import glu


class AttentiveTransformer(keras.layers.Layer):
    """
    Attentive Transformer layer for mask generation
    as proposed by Sercan et al. in TabNet paper.
    It applies a Dense layer followed by a BatchNormalization layer
    followed by a GLU activation layer.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        TODO
    """
    def __init__(self,
                 num_features,
                 batch_momentum=0.9,
                 virtual_batch_size: int = None,
                 relaxation_factor=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.relaxation_factor = relaxation_factor

        self.dense = keras.layers.Dense(self.num_features, use_bias=False)
        self.batch_norm = keras.layers.BatchNormalization(momentum=batch_momentum,
                                                          virtual_batch_size=self.virtual_batch_size)

    def call(self, inputs, prior_scales=None):
        # We need the batch_size and inputs_dimensions to initialize prior scale,
        # we can get them when attentive transformer first gets called.
        outputs = self.dense(inputs)
        outputs = self.batch_norm(outputs)
        outputs *= prior_scales
        outputs = tfa.activations.sparsemax(outputs)
        return outputs


class FeatureTransformer(layers.Layer):
    """
    Feature Transformer layer is used in constructing the FeatureTransformer block
    as proposed by Sercan et al. in TabNet paper.
    It applies a Dense layer followed by a BatchNormalization layer
    followed by a GLU activation layer.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
    """
    def __init__(self,
                 units,
                 batch_momentum=0.9,
                 virtual_batch_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size

        self.dense = keras.layers.Dense(self.units * 2, use_bias=False)
        self.norm = keras.layers.BatchNormalization(momentum=self.batch_momentum,
                                                    virtual_batch_size=virtual_batch_size)
        self.glu = GLU(self.units)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = self.glu(x)
        return x


class FeatureTransformerBlock(layers.Layer):
    """
    Feature Transformer block as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
    """
    def __init__(self,
                 units,
                 batch_momentum=0.9,
                 virtual_batch_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size

        self.feat_transform_1 = FeatureTransformer(self.units, self.batch_momentum, self.virtual_batch_size)
        self.feat_transform_2 = FeatureTransformer(self.units, self.batch_momentum, self.virtual_batch_size)
        self.feat_transform_3 = FeatureTransformer(self.units, self.batch_momentum, self.virtual_batch_size)
        self.feat_transform_4 = FeatureTransformer(self.units, self.batch_momentum, self.virtual_batch_size)

    def call(self, inputs):
        x = self.feat_transform_1(inputs)
        residual = x
        x = self.feat_transform_2(x)
        # Normalized residual:
        # Normalization with sqrt(0.5) helps to stabilize learning by ensuring
        # that the variance throughout the network does not change dramatically
        x = (x + residual) * tf.math.sqrt(0.5)
        residual = x
        x = self.feat_transform_3(x)
        x = (x + residual) * tf.math.sqrt(0.5)
        residual = x
        x = self.feat_transform_4(x)
        x = (x + residual) * tf.math.sqrt(0.5)
        return x


class Encoder(layers.Layer):
    """
    Encoder as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        output_dim: Output dimensionality for the decision step
        relaxation_factor: When = 1, a feature is enforced to be used only at one decision step
                        and as it increases, more flexibility is provided to use a feature at multiple decision steps.
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
        epsilon: Epsilon is a small number for numerical stability.
    """

    def __init__(self,
                 feature_transformer_dim,
                 decision_step_output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 epsilon,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_output_dim = decision_step_output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

        self.inputs_norm = keras.layers.BatchNormalization(momentum=self.batch_momentum)
        self.feature_transformer_block = FeatureTransformerBlock(units=self.feature_transformer_dim,
                                                                 batch_momentum=self.batch_momentum,
                                                                 virtual_batch_size=self.virtual_batch_size,
                                                                 name="feature_transformer_block"
                                                                 )
        self.feature_importances_per_sample = []
        self.relu = keras.layers.ReLU()

    def build(self, input_shape):
        # Number of features or number of columns is equivalent to the input_dimension i.e. the last dimension
        num_features = input_shape[1]

        # Attentive transformer -- used for creating mask
        # We initialize this layer in the build method here
        # since we need to know the num_features for its initialization
        self.attentive_transformer = AttentiveTransformer(num_features=num_features,
                                                          batch_momentum=self.batch_momentum,
                                                          virtual_batch_size=self.virtual_batch_size,
                                                          relaxation_factor=self.relaxation_factor,
                                                          name="attentive_transformer")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Initializes decision-step dependent variables
        outputs_aggregated = tf.zeros(shape=tf.shape(inputs))
        mask_values = tf.zeros(shape=tf.shape(inputs))
        # Aggregated mask values are used for explaining feature importance
        # Here we'll refer to them as feature_importances_per_sample since
        # that's a more descriptive name
        # For more details, read the `Interpretability` section on page 5
        # of the paper "TabNet: Attentive Interpretable Tabular Learning"
        feature_importances_per_sample = tf.zeros(shape=tf.shape(inputs))
        total_entropy = 0.

        # Prior scales `P` indicate how much a particular feature has been used previously
        # P[i] = ((Pi)j=1 to i)(gamma (gamma - M[j])
        # where `Pi` is the Product Greek letter
        # gamma is the relaxation factor, when gamma = 1, a feature is enforced,
        # to be used only at one decision step and as gamma increases, more
        # flexibility is provided to use a feature at multiple decision steps.
        # P[0] is initialized as all ones, `1`B×D, without any prior
        # on the masked features.
        prior_scales = tf.ones(tf.shape(inputs))

        normalized_inputs = self.inputs_norm(inputs)
        masked_features = normalized_inputs

        for n_step in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below
            x = self.feature_transformer_block(masked_features)

            if n_step > 0 or self.num_decision_steps == 1:
                decision_step_outputs = self.relu(x[:, :self.decision_step_output_dim])
                # Decision aggregation.
                outputs_aggregated += decision_step_outputs
                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_step_outputs,
                                          axis=1,
                                          keepdims=True)

                # To prevent division by zero, we introduce this conditional
                # and only divide by (num_decision_steps - 1) if num_decision_steps > 1
                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)

                feature_importances_per_sample += (mask_values * scale_agg)
                self.feature_importances_per_sample.extend(feature_importances_per_sample)

            features_for_attentive_transformer = (x[:, self.decision_step_output_dim:])

            if n_step < self.num_decision_steps - 1:
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use
                mask_values = self.attentive_transformer(features_for_attentive_transformer,
                                                         prior_scales=prior_scales)
                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of coefficients
                prior_scales *= (self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature selection
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(-mask_values * tf.math.log(mask_values + self.epsilon), axis=1)
                ) / (self.num_decision_steps - 1)
                entropy_loss = total_entropy

                # Feature Selection
                masked_features = tf.multiply(mask_values, normalized_inputs)
            else:
                entropy_loss = 0.

        # Add entropy loss
        self.add_loss(entropy_loss)

        return self.output_aggregated
