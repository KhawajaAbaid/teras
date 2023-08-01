import tensorflow as tf
from tensorflow import keras
from teras.layers.tabnet.tabnet_feature_transformer import TabNetFeatureTransformer
from teras.layers.tabnet.tabnet_attentive_transformer import TabNetAttentiveTransformer


@keras.saving.register_keras_serializable("teras.layers.tabnet")
class TabNetEncoder(keras.layers.Layer):
    """
    TabNetEncoder layer based on the TabNet architecture
    proposed by Sercan et al. in the TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: ``int``,
            The dimensionality of the original input dataset,
            or the number of features in the original input dataset.

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
    """
    def __init__(self,
                 data_dim: int,
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
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_output_dim = decision_step_output_dim
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.residual_normalization_factor = residual_normalization_factor
        self.epsilon = epsilon

        self.inputs_norm = keras.layers.BatchNormalization(momentum=batch_momentum)

        # To avoid shared layers errors, we should clear them before instantiating
        # the TabNetEncoder layer.
        TabNetFeatureTransformer.reset_shared_layers()
        self.features_transformers_per_step = [TabNetFeatureTransformer(
            units=self.feature_transformer_dim * 2,
            num_shared_layers=self.num_shared_layers,
            num_decision_dependent_layers=self.num_decision_dependent_layers,
            batch_momentum=self.batch_momentum,
            virtual_batch_size=self.virtual_batch_size,
            residual_normalization_factor=self.residual_normalization_factor,
            name=f"step_{i}_feature_transformer"
        )
            for i in range(self.num_decision_steps)
        ]

        # Attentive transformer -- used for creating mask
        self.attentive_transformers_per_step = [TabNetAttentiveTransformer(data_dim=self.data_dim,
                                                                           batch_momentum=self.batch_momentum,
                                                                           virtual_batch_size=self.virtual_batch_size,
                                                                           relaxation_factor=self.relaxation_factor,
                                                                           name=f"step_{i}_attentive_transformer")
                                                for i in range(self.num_decision_steps)
                                                ]
        self.feature_importances_per_sample = []
        self.relu = keras.layers.ReLU()

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        # Initializes decision-step dependent variables
        outputs_aggregated = tf.zeros(shape=(batch_size, self.decision_step_output_dim))
        # we only pass mask alongside inputs when we're pretaining
        # As the paper says in the Self-Supervised section, we  initialize
        # mask_values (P) to be 1-S during pretraining.
        # we create this S in the TabNetPretrainer class and pass it from there
        if mask is not None:
            mask_values = mask
        else:
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
                self.feature_importances_per_sample.append(feature_importances_per_sample)

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

    def get_config(self):
        config = super().get_config()
        new_config = {'data_dim': self.data_dim,
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
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        data_dim = config.pop("data_dim")
        return cls(data_dim=data_dim, **config)
