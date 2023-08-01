import tensorflow as tf
from tensorflow import keras
from teras.layers.tabnet import TabNetFeatureTransformer


@keras.saving.register_keras_serializable("teras.layers.tabnet")
class TabNetDecoder(keras.layers.Layer):
    """
    TabNetDecoder as proposed by Sercan et al. in TabNet paper.

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

        # To be able to share the `shared_layers` across instances, we introduced
        # a class attribute called `shared_layers` in the FeatureTransformer class
        # BUT here's the problem,
        # even though we now want to create new shared layers separately for our Decoder, it won't ... unless...
        # we do this
        TabNetFeatureTransformer.reset_shared_layers()
        # Why? Because each time we create `TabNetFeatureTransformer` instance, it first checks if `shared_layers`
        # attribute is None or not, if it's None it will create new shared layers, otherwise it won't.

        for i in range(self.num_decision_steps):
            self.features_transformers_per_step.append(TabNetFeatureTransformer(
                                                        units=self.feature_transformer_dim,
                                                        num_shared_layers=self.num_shared_layers,
                                                        num_decision_dependent_layers=self.num_decision_dependent_layers,
                                                        batch_momentum=self.batch_momentum,
                                                        virtual_batch_size=self.virtual_batch_size,
                                                        residual_normalization_factor=self.residual_normalization_factor,
                                                        name=f"step_{i}_feature_transformer"
                                                        ))
            self.projection_layers_per_step.append(keras.layers.Dense(self.data_dim))

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
