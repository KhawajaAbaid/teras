import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
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


class FeatureTransformerBlock(layers.Layer):
    """
    Feature Transformer block layer is used in constructing the FeatureTransformer
    as proposed by Sercan et al. in TabNet paper.
    It applies a Dense layer followed by a BatchNormalization layer
    followed by a GLU activation layer.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
        use_residual_normalization
    """
    def __init__(self,
                 units,
                 batch_momentum=0.9,
                 virtual_batch_size=None,
                 use_residual_normalization: bool = True,
                 residual_normalization_factor=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        # The official implementation says,
        # feature_dimensionality (here `units`) is the dimensionality of the hidden representation in feature
        # transformation block. In which, each layer first maps the representation to a
        # 2*feature_dim-dimensional output and half of it is used to determine the
        # non-linearity of the GLU activation where the other half is used as an
        # input to GLU, and eventually feature_dim-dimensional output is
        # transferred to the next layer.
        self.units = units
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.use_residual_normalization = use_residual_normalization
        self.residual_normalization_factor = residual_normalization_factor
        self.dense = keras.layers.Dense(self.units * 2, use_bias=False)
        self.norm = keras.layers.BatchNormalization(momentum=self.batch_momentum,
                                                    virtual_batch_size=virtual_batch_size)

        self.glu = GLU(self.units)
        self.add = keras.layers.Add()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = self.glu(x)
        if self.use_residual_normalization:
            x = self.add([x, inputs]) * tf.math.sqrt(self.residual_normalization_factor)
        return x


class FeatureTransformer(layers.Layer):
    """
    Feature Transformer as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: Number of hidden units to use in Fully Connected (Dense) layer
        batch_momentum: Momentum value to use for BatchNormalization layer
        virtual_batch_size: Batch size to use for virtual_batch_size parameter in BatchNormalization layer
    """
    shared_layers = None
    def __init__(self,
                 units,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 batch_momentum=0.9,
                 virtual_batch_size=None,
                 residual_normalization_factor=0.5,
                 **kwargs):

        if num_shared_layers == 0 and num_decision_dependent_layers == 0:
            raise ValueError("You both can't be zero (TODO add more specific error msg)")

        super().__init__(**kwargs)
        self.units = units
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.residual_normalization_factor = residual_normalization_factor

        # Since shared layers should be accessible across instances so we make them as class attributes
        # and use class method to instantiate them, the first time an instance of this FeatureTransformer
        # class is made.
        if self.shared_layers is None and num_shared_layers > 0:
            self._create_shared_layers(num_shared_layers,
                                       units,
                                       batch_momentum=batch_momentum,
                                       virtual_batch_size=virtual_batch_size,
                                       residual_normalization_factor=residual_normalization_factor)

        self.inner_block = models.Sequential(name="inner_block")

        self.inner_block.add(self.shared_layers)

        if self.num_decision_dependent_layers > 0:
            decision_dependent_layers = models.Sequential(name=f"decision_dependent_layers")
            for j in range(self.num_decision_dependent_layers):
                if j == 0 and self.num_shared_layers == 0:
                    # then it means that this is the very first layer in the Feature Transformer
                    # because no shard layers exist.
                    # hence we shouldn't use residual normalization
                    use_residual_normalization = False
                else:
                    use_residual_normalization = True
                decision_dependent_layers.add(FeatureTransformerBlock(
                                                units=self.units,
                                                batch_momentum=self.batch_momentum,
                                                virtual_batch_size=self.virtual_batch_size,
                                                use_residual_normalization=use_residual_normalization,
                                                name=f"decision_dependent_layer_{j}")
                                                )

            self.inner_block.add(decision_dependent_layers)

    @classmethod
    def _create_shared_layers(cls, num_shared_layers, units, **kwargs):
        # The outputs of first layer in feature transformer isn't residual normalized
        # so we use a pointer to know which layer is first.
        # Important to keep in mind that shared layers ALWAYS precede the decision dependent layers
        # BUT we want to allow user to be able to use NO shared layers at all, i.e. 0.
        # Hence, only if the `num_shared_layers` is 0, should the first dependent layer for each step
        # be treated differently.
        # Initialize the block of shared layers
        cls.shared_layers = models.Sequential(name="shared_layers")
        for i in range(num_shared_layers):
            if i == 0:
                # First layer should not use residual normalization
                use_residual_normalization = False
            else:
                use_residual_normalization = True
            cls.shared_layers.add(FeatureTransformerBlock(units=units,
                                                          use_residual_normalization=use_residual_normalization,
                                                          name=f"shared_layer_{i}",
                                                          **kwargs))

    def call(self, inputs):
        outputs = self.inner_block(inputs)
        return outputs


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
                 feature_transformer_dim: int = 32,
                 decision_step_output_dim: int = 32,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 num_decision_steps: int = 5,
                 relaxation_factor=1.5,
                 batch_momentum=0.7,
                 virtual_batch_size: int = 16,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_output_dim = decision_step_output_dim
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

        self.inputs_norm = keras.layers.BatchNormalization(momentum=self.batch_momentum)

        self.features_transformers_per_step = [FeatureTransformer(
                                            units=self.feature_transformer_dim,
                                            num_shared_layers=self.num_shared_layers,
                                            num_decision_dependent_layers=self.num_decision_dependent_layers,
                                            batch_momentum=self.batch_momentum,
                                            virtual_batch_size=self.virtual_batch_size,
                                            name=f"step_{i}_feature_transformer"
                                        )
                                        for i in range(self.num_decision_steps)
                                    ]

        self.feature_importances_per_sample = []
        self.relu = keras.layers.ReLU()

    def build(self, input_shape):
        # Number of features or number of columns is equivalent to the input_dimension i.e. the last dimension
        num_features = input_shape[1]

        # Attentive transformer -- used for creating mask
        # We initialize this layer in the build method here
        # since we need to know the num_features for its initialization
        self.attentive_transformers_per_step = [AttentiveTransformer(num_features=num_features,
                                                                     batch_momentum=self.batch_momentum,
                                                                     virtual_batch_size=self.virtual_batch_size,
                                                                     relaxation_factor=self.relaxation_factor,
                                                                     name=f"step_{i}_attentive_transformer")
                                                for i in range(self.num_decision_steps)
                                                ]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Initializes decision-step dependent variables
        outputs_aggregated = tf.zeros(shape=(batch_size, self.decision_step_output_dim))
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

        for step_num in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below
            x = self.features_transformers_per_step[step_num](masked_features)

            if step_num > 0 or self.num_decision_steps == 1:
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

            if step_num < self.num_decision_steps - 1:
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use
                mask_values = self.attentive_transformers_per_step[step_num](features_for_attentive_transformer,
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

        return outputs_aggregated
