import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
from teras.layers import GLU
from teras.layers.common.head import (RegressionHead as BaseRegressionHead,
                                      ClassificationHead as BaseClassificationHead)
from typing import Union, List, Tuple

LIST_OR_TUPLE = Union[List[int], Tuple[int]]
LAYER_OR_STR = Union[keras.layers.Layer, str]


class AttentiveTransformer(layers.Layer):
    """
    Attentive Transformer layer for mask generation
    as proposed by Sercan et al. in TabNet paper.
    It applies a Dense layer followed by a BatchNormalization layer
    followed by a GLU activation layer.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        num_features: `int`, Number of features in the input dataset.
        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
        relaxation_factor: `float`, default 1.5, Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on the performance.
            Typically, a larger value for `num_decision_steps` favors for a larger `relaxation_factor`.

    """
    def __init__(self,
                 num_features,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 relaxation_factor: float = 1.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.relaxation_factor = relaxation_factor

        self.dense = layers.Dense(self.num_features, use_bias=False)
        self.batch_norm = layers.BatchNormalization(momentum=batch_momentum,
                                                    virtual_batch_size=self.virtual_batch_size)

    def call(self, inputs, prior_scales=None):
        # We need the batch_size and inputs_dimensions to initialize prior scale,
        # we can get them when attentive transformer first gets called.
        outputs = self.dense(inputs)
        outputs = self.batch_norm(outputs)
        outputs *= prior_scales
        outputs = tfa.activations.sparsemax(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'num_features': self.num_features,
                      'batch_momentum': self.batch_momentum,
                      'virtual_batch_size': self.virtual_batch_size,
                      'relaxation_factor': self.relaxation_factor,
                      }
        config.update(new_config)
        return config


class FeatureTransformerBlock(layers.Layer):
    """
    Feature Transformer block layer is used in constructing the FeatureTransformer
    as proposed by Sercan et al. in TabNet paper.
    It applies a Dense layer followed by a BatchNormalization layer
    followed by a GLU activation layer.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: `int`, default 32, the dimensionality of the hidden
            representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.        batch_momentum: Momentum value to use for BatchNormalization layer
        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
            This is typically much smaller than the `batch_size` used for training.
        residual_normalization_factor: `float`, default 0.5, In the feature transformer, except for the
            layer, every other layer utilizes normalized residuals, where `residual_normalization_factor`
            determines the scale of normalization.
        use_residual_normalization: `bool`, default True, Whether to use residual normalization.
            According to the default architecture, every layer uses residual normalization EXCEPT
            for the very first layer.
    """
    def __init__(self,
                 units: int,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 use_residual_normalization: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.use_residual_normalization = use_residual_normalization
        self.residual_normalization_factor = residual_normalization_factor
        self.dense = layers.Dense(self.units * 2, use_bias=False)
        self.norm = layers.BatchNormalization(momentum=self.batch_momentum,
                                              virtual_batch_size=virtual_batch_size)

        self.glu = GLU(self.units)
        self.add = layers.Add()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.norm(x)
        x = self.glu(x)
        if self.use_residual_normalization:
            x = self.add([x, inputs]) * tf.math.sqrt(self.residual_normalization_factor)
        return x

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'batch_momentum': self.batch_momentum,
                      'virtual_batch_size': self.virtual_batch_size,
                      'residual_normalization_factor': self.residual_normalization_factor,
                      'use_residual_normalization': self.use_residual_normalization,
                      }
        config.update(new_config)
        return config


class FeatureTransformer(layers.Layer):
    """
    Feature Transformer as proposed by Sercan et al. in TabNet paper.
    It is made up of FeatureTransformerBlock building blocks.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        units: `int`, default 32, the dimensionality of the hidden
            representation in feature transformation block.
            Each layer first maps the representation to a `2 * feature_transformer_dim`
            output and half of it is used to determine the
            non-linearity of the GLU activation where the other half is used as an
            input to GLU, and eventually `feature_transformer_dim` output is
            transferred to the next layer.
        num_shared_layers: `int`, default 2. Number of shared layers to use in the Feature Transformer.
            These shared layers are `shared` across decision steps.
        num_decision_dependent_layers: `int`, default 2. Number of decision dependent layers to use in
            the Feature Transformer. In simple words, `num_decision_dependent_layers` are created
            for each decision step in the `num_decision_steps`.
            For instance, if `num_decision_steps = 5` and  `num_decision_dependent_layers = 2`
            then 10 layers will be created, 2 for each decision step.
        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
            This is typically much smaller than the `batch_size` used for training.
        residual_normalization_factor: `float`, default 0.5, In the feature transformer, except for the
            layer, every other layer utilizes normalized residuals, where `residual_normalization_factor`
            determines the scale of normalization.
    """
    shared_layers = None

    def __init__(self,
                 units,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 **kwargs):

        if num_shared_layers == 0 and num_decision_dependent_layers == 0:
            raise ValueError("Feature Transformer requires at least one of either shared or decision depenedent layers."
                             " But both `num_shared_layers` and `num_decision_dependent_layers` were passed a 0 value.")

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

    def get_config(self):
        config = super().get_config()
        new_config = {'units': self.units,
                      'num_shared_layers': self.num_shared_layers,
                      'num_decision_dependent_layers': self.num_decision_dependent_layers,
                      'batch_momentum': self.batch_momentum,
                      'virtual_batch_size': self.virtual_batch_size,
                      'residual_normalization_factor': self.residual_normalization_factor,
                      }
        config.update(new_config)
        return config


class Encoder(layers.Layer):
    """
    Encoder as proposed by Sercan et al. in TabNet paper.

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
        num_features: `int`, Number of features in the dataset.
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
                 num_features: int = None,
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
        self.residual_normalization_factor = residual_normalization_factor
        self.epsilon = epsilon
        self.num_features = num_features

        self.inputs_norm = layers.BatchNormalization(momentum=batch_momentum)

        self.features_transformers_per_step = [FeatureTransformer(
            units=self.feature_transformer_dim,
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
        # We initialize this layer in the build method here
        # since we need to know the num_features for its initialization
        self.attentive_transformers_per_step = [AttentiveTransformer(num_features=self.num_features,
                                                                     batch_momentum=self.batch_momentum,
                                                                     virtual_batch_size=self.virtual_batch_size,
                                                                     relaxation_factor=self.relaxation_factor,
                                                                     name=f"step_{i}_attentive_transformer")
                                                for i in range(self.num_decision_steps)
                                                ]

        self.feature_importances_per_sample = []
        self.relu = layers.ReLU()

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
        new_config = {'feature_transformer_dim': self.feature_transformer_dim,
                      'decision_step_output_dim': self.decision_step_output_dim,
                      'num_decision_steps': self.num_decision_steps,
                      'num_shared_layers': self.num_shared_layers,
                      'num_decision_dependent_layers': self.num_decision_dependent_layers,
                      'relaxation_factor': self.relaxation_factor,
                      'batch_momentum': self.batch_momentum,
                      'virtual_batch_size': self.virtual_batch_size,
                      'residual_normalization_factor': self.residual_normalization_factor,
                      'epsilon': self.epsilon,
                      'num_features': self.num_features,
                      }
        config.update(new_config)
        return config


class Decoder(layers.Layer):
    """
    Decoder as proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: Dimensionality of the original dataset - or the total number of
            features in the original dataset.
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
        batch_momentum: `float`, default 0.9, Momentum value to use for BatchNormalization layer.
        virtual_batch_size: `int`, default 64, Batch size to use for `virtual_batch_size`
            parameter in BatchNormalization layer.
            This is typically much smaller than the `batch_size` used for training.
        residual_normalization_factor: `float`, default 0.5, In the feature transformer, except for the
            layer, every other layer utilizes normalized residuals, where `residual_normalization_factor`
            determines the scale of normalization.
    """
    def __init__(self,
                 data_dim: int,
                 feature_transformer_dim: int = 32,
                 decision_step_output_dim: int = 32,
                 num_decision_steps: int = 5,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 batch_momentum: float = 0.9,
                 virtual_batch_size: int = 64,
                 residual_normalization_factor: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_output_dim = decision_step_output_dim
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.num_decision_steps = num_decision_steps
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.residual_normalization_factor = residual_normalization_factor

        self.features_transformers_per_step = []
        self.projection_layers_per_step = []

        # OKAY LISTEN: To be able to share the `shared_layers` across instances, we introduced
        # a class attribute called `shared_layers` in the FeatureTransformer class BUT here's the problem,
        # even though we now want to create shared layers separately for our Decoder, it won't ... unless...
        # we do this
        FeatureTransformer.shared_layers = None
        # Why? Because each time we create FeatureTransformer instance, it first checks if `shared_layers`
        # attribute is None or not, if it's None it will create new shared layers, otherwise it won't.

        for i in range(self.num_decision_steps):
            self.features_transformers_per_step.append(FeatureTransformer(
                                                        units=self.feature_transformer_dim,
                                                        num_shared_layers=self.num_shared_layers,
                                                        num_decision_dependent_layers=self.num_decision_dependent_layers,
                                                        batch_momentum=self.batch_momentum,
                                                        virtual_batch_size=self.virtual_batch_size,
                                                        residual_normalization_factor=self.residual_normalization_factor,
                                                        name=f"step_{i}_feature_transformer"
                                                        ))
            self.projection_layers_per_step.append(layers.Dense(self.data_dim))

    def call(self, inputs, mask=None):
        """
        Args:
            inputs: Encoded representations.

        Returns:

        """
        batch_size = tf.shape(inputs)[0]
        reconstructed_features = tf.zeros(shape=(batch_size, self.data_dim))

        for i in range(self.num_decision_steps):
            feat_output = self.features_transformers_per_step[i](inputs)
            reconstructed_features += self.projection_layers_per_step[i](feat_output)

        # The paper says,
        # the decoder’s last FC (dense) layer is multiplied with S (binary mask indicating which features are missing)
        # to output the unknown features.
        if mask is not None:
            reconstructed_features *= mask

        return reconstructed_features

    def get_config(self):
        config = super().get_config()
        new_config = {'data_dim': self.data_dim,
                      'feature_transformer_dim': self.feature_transformer_dim,
                      'decision_step_output_dim': self.decision_step_output_dim,
                      'num_decision_steps': self.num_decision_steps,
                      'num_shared_layers': self.num_shared_layers,
                      'num_decision_dependent_layers': self.num_decision_dependent_layers,
                      'batch_momentum': self.batch_momentum,
                      'virtual_batch_size': self.virtual_batch_size,
                      'residual_normalization_factor': self.residual_normalization_factor,
                      }
        config.update(new_config)
        return config


class RegressionHead(BaseRegressionHead):
    """
    Regression head for the TabNet Regressor architecture.

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs to predict.
        units_values: `List[int] | Tuple[int]`, default `None`,
            If specified, for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the RegressionHead.
        activation_hidden: default `None`,
            Activation function to use in hidden dense layers.
        normalization: `Layer | str`, default `None`,
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 units_values: LIST_OR_TUPLE = None,
                 activation_hidden=None,
                 normalization: LAYER_OR_STR = None,
                 **kwargs):
        super().__init__(num_outputs=num_outputs,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         normalization=normalization,
                         **kwargs)


class ClassificationHead(BaseClassificationHead):
    """
    Classification head for TabNet Classifier model.

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        units_values: `List[int] | Tuple[int]`, default `None`,
            If specified, for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the ClassificationHead.
        activation_hidden: default `None`,
            Activation function to use in hidden dense layers.
        activation_out: default `None`,
            Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.
        normalization: `Layer | str`, default `None`,
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_classes: int = 2,
                 units_values: LIST_OR_TUPLE = None,
                 activation_hidden=None,
                 activation_out=None,
                 normalization: LAYER_OR_STR = None,
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out,
                         normalization=normalization,
                         **kwargs)
