import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from teras.layers import GLU

class FeatureTransformerBlock(layers.Layer):
    """
    Feature Transformer block is used in constructing the FeatureTransformer layer
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
        # currently only batch normalization is supported as proposed in the paper
        self.norm = keras.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=virtual_batch_size)
        self.glu = GLU(self.units)

    def call(self,
             inputs,
             *args,
             **kwargs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = self.glu(x)
        return x

class FeatureTransformer(layers.Layer):
    """
    Feature Transformer layer as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
    """
    def __init__(self, units,
                 batch_momentum=0.9,
                 virtual_batch_size=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size

        self.feat_transform_1 = FeatureTransformerBlock(self.units, self.batch_momentum, self.virtual_batch_size)
        self.feat_transform_2 = FeatureTransformerBlock(self.units, self.batch_momentum, self.virtual_batch_size)
        self.feat_transform_3 = FeatureTransformerBlock(self.units, self.batch_momentum, self.virtual_batch_size)
        self.feat_transform_4 = FeatureTransformerBlock(self.units, self.batch_momentum, self.virtual_batch_size)

    def call(self, inputs, *args, **kwargs):
        x = self.feat_transform_1(inputs)
        residual = x
        x = self.feat_transform_2(x)
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

    def __init__(self, units,
                 output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 epsilon,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

        self.inputs_norm = keras.layers.BatchNormalization(momentum=self.batch_momentum)
        self.feature_transformer = FeatureTransformer(self.units, self.batch_momentum, self.virtual_batch_size)

        # Feature Masks Transformer Layers
        # We initialize the dense layer in the build method below since we need to know the num_features to create that layer
        # Although we initialize the batch normalization and sparsemax layers here to stay in accordance with the best practices
        self.feature_masks_transformer_norm = keras.layers.BatchNormalization(momentum=self.batch_momentum,
                                                                              virtual_batch_size=self.virtual_batch_size)
        self.feature_masks_transformer_sparsemax = tfa.layers.Sparsemax()

    def build(self, input_shape):
        # Number of features or number of columns is equivalent to the input_dimension i.e. the last dimension
        self.num_features = input_shape[-1]

        # Feature masks transformer layers (cont.)
        # We initialize this layer in the build method here
        # since we need to know the num_features for its initialization
        self.feature_masks_transformer_dense = keras.layers.Dense(self.num_features)

    def call(self,
             inputs,
             *args,
             **kwargs):
        batch_size = tf.shape(inputs)[0]
        # Initializes decision-step dependent variables
        self.output_aggregated = tf.zeros(shape=(batch_size, self.output_dim))
        self.mask_values = tf.zeros(shape=(batch_size, self.num_features))
        self.aggregated_mask_values = tf.zeros(shape=(batch_size, self.num_features))
        self.complemantary_aggregated_mask_values = tf.ones(shape=(batch_size, self.num_features))
        self.total_entropy = 0.

        normalized_inputs = self.inputs_norm(inputs)
        masked_features = normalized_inputs



        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below
            x = self.feature_transformer(masked_features)

            if ni > 0 or self.num_decision_steps == 1:
                decision_out = tf.nn.relu(x[:, :self.output_dim])
                # Decision aggregation.
                self.output_aggregated += decision_out
                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)

                self.aggregated_mask_values += self.mask_values * scale_agg

            features_for_coef = (x[:, self.output_dim:])

            if ni < self.num_decision_steps - 1:
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use
                self.mask_values = self.feature_masks_transformer_dense(features_for_coef)
                self.mask_values = self.feature_masks_transformer_norm(self.mask_values)
                self.mask_values *= self.complemantary_aggregated_mask_values
                self.mask_values = self.feature_masks_transformer_sparsemax(self.mask_values)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of coefficients
                self.complemantary_aggregated_mask_values *= (
                    self.relaxation_factor - self.mask_values)

                # Entropy is used to penalize the amount of sparsity in feature selection
                self.total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -self.mask_values * tf.math.log(self.mask_values + self.epsilon),
                        axis=1)) / (
                            self.num_decision_steps - 1)

                entropy_loss = self.total_entropy

                # Feature Selection
                masked_features = tf.multiply(self.mask_values, normalized_inputs)

                # Visualization of the feature selection mask at the decision step ni
                tf.summary.image(
                    "Mask for step" + str(ni),
                    tf.expand_dims(tf.expand_dims(self.mask_values, 0), 3), max_outputs=1)
            else:
                entropy_loss = 0.

        # Add entropy loss
        self.add_loss(entropy_loss)

        # Visualization of the aggregated feature importances
        tf.summary.image(
            "Aggregated Mask",
            tf.expand_dims(tf.expand_dims(self.aggregated_mask_values, 0), 3), max_outputs=1)

        return self.output_aggregated
