import keras
from keras import ops

from teras._src.layers.layer_list import LayerList
from teras._src.layers.tabnet.attentive_transformer import \
    TabNetAttentiveTransformer
from teras._src.layers.tabnet.feature_transformer import \
    TabNetFeatureTransformer
from teras._src.api_export import teras_export


@teras_export("teras.models.TabNetEncoderBackbone")
class TabNetEncoderBackbone(keras.Model):
    """
    TabNetEncoder proposed by Arik et al. in the
    "TabNet: Attentive Interpretable Tabular Learning" paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        input_dim: int, The dimensionality of the input dataset.
        feature_transformer_dim: int, the dimensionality of the hidden
            representation in feature transformation block.
            The dense layer inside the `TabNetFeatureTransformer` first
            maps  the inputs to `feature_transformer_dim` * 2 dimension
            hidden representations and later the glu activation
            maps the hidden representations to
            `feature_transformer_dim` dimensions.
        decision_step_dim: int, the dimensionality of output at each
            decision step, which is later mapped to the final
            classification or regression output.
            It is recommended to keep `decision_step_dim` and
            `feature_transformer_dim` equal to each other.
            Adjusting these two parameters values is a good way of
            obtaining a tradeoff between performance and complexity.
        num_decision_steps: int, the number of sequential decision steps.
            For most datasets a value in the range [3, 10] is optimal.
            If there are more informative features in the dataset,
            the value tends to be higher. That said, a very high value
            of `num_decision_steps` may suffer from overfitting.
            Defaults to 5.
        num_shared_layers: int, Number of shared layers to use in the
            ``TabNetFeatureTransformer``.
            These shared layers are shared across decision steps.
            Defaults to 2.
        num_decision_dependent_layers: int, number of decision dependent
            layers to use in the `TabNetFeatureTransformer`.
            In simple words, `num_decision_dependent_layers` are created
            for each decision step in the `num_decision_steps`.
            For instance, if `num_decision_steps = 5` and
            `num_decision_dependent_layers = 2`
            then 10 layers will be created, 2 for each decision step.
        relaxation_factor: float, relaxation factor that promotes the
            reuse of each feature at different decision steps. When it
            is 1, a feature is enforced to be used only at one decision
            step and as it increases, more flexibility is provided to
            use a feature at multiple decision steps.
            An optimal value of relaxation_factor can have a major role on
            the performance. Typically, a larger value for
            `num_decision_steps` favors for a larger `relaxation_factor`.
        batch_momentum: float, momentum value to use for
            `BatchNormalization` layer.
            Defaults to 0.9
        epsilon: float, epsilon is a small number for numerical stability
            during the computation of entropy loss.
            Defaults to 0.00001
        reuse_shared_layers: bool, whether to reset shared layers of the
            `TabNetFeatureTransformer` layer.
            Although we want to use the same shared layers across
            multiple instances of `TabNetFeatureTransformer` but we may
            not want to use the same shared layers across different
            `TabNetEncoder` instances.
            Defaults to `True`.
    """
    def __init__(self,
                 input_dim: int,
                 feature_transformer_dim: int,
                 decision_step_dim: int,
                 num_decision_steps: int = 5,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 5,
                 relaxation_factor: float = 1.5,
                 batch_momentum: float = 0.9,
                 epsilon: float = 1e-5,
                 reset_shared_layers: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_dim = decision_step_dim
        self.num_decision_steps = num_decision_steps
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        self.reset_shared_layers = reset_shared_layers

        if self.reset_shared_layers:
            TabNetFeatureTransformer.reset_shared_layers()

        self.feature_transformers = LayerList([
            TabNetFeatureTransformer(
                hidden_dim=self.feature_transformer_dim,
                num_shared_layers=self.num_decision_steps,
                num_decision_dependent_layers=self.num_decision_dependent_layers,
                batch_momentum=self.batch_momentum,
                name=f"encoder_feature_transformer_{i}"
            )
            for i in range(self.num_decision_steps)
        ],
            sequential=False,
            name="encoder_feature_transformers"
        )
        self.attentive_transformers = LayerList([
            TabNetAttentiveTransformer(
                data_dim=input_dim,
                batch_momentum=self.batch_momentum,
                name=f"encoder_attentive_transformer_{i}"
            )
            for i in range(self.num_decision_steps - 1)
        ],
            sequential=False,
            name="encoder_attentive_transformers"
        )
        self.batch_norm = keras.layers.BatchNormalization(
            momentum=self.batch_momentum
        )

    def build(self, input_shape):
        input_shape = tuple(input_shape)
        self._input_shape = input_shape
        self.batch_norm.build(input_shape)
        self.feature_transformers.build(input_shape)
        input_shape = input_shape[:-1] + (
            self.feature_transformer_dim - self.decision_step_dim,)
        self.attentive_transformers.build(input_shape)

    def call(self, inputs, mask=None):
        normalized_inputs = self.batch_norm(inputs)
        batch_size = ops.shape(inputs)[0]
        decision_out_aggregated = ops.zeros(
            (batch_size, self.decision_step_dim))
        # During pretraining, we pass mask alongside inputs to the encoder
        masked_features = normalized_inputs
        if mask is not None:
            mask_values = mask
        else:
            mask_values = ops.zeros_like(inputs)
        aggregated_mask_values = ops.zeros_like(inputs)
        # Prior scales denote how much a feature has been used previously
        prior_scales = ops.ones_like(inputs)
        total_entropy = 0.

        for d_step in range(self.num_decision_steps):
            feat_transformer_out = self.feature_transformers[d_step](
                masked_features)
            if d_step > 0 or self.num_decision_steps == 1:
                decision_out = ops.relu(
                    feat_transformer_out[:, :self.decision_step_dim])
                decision_out_aggregated += decision_out

                # Aggregated masks are used for the visualization of the
                # feature importance attributes.
                scale = ops.sum(
                    decision_out, axis=-1, keepdims=True) / (
                                self.num_decision_steps - 1)
                aggregated_mask_values += mask_values / scale

            the_other_output_half = feat_transformer_out[:, self.decision_step_dim:]

            if d_step < self.num_decision_steps - 1:
                mask = self.attentive_transformers[d_step](
                    the_other_output_half,
                    prior_scales)
                # Relaxation factor controls the amount of reuse of
                # features between different decision blocks and updated
                # with the values of coefficients.
                prior_scales = prior_scales * (self.relaxation_factor -
                                               mask)

                # Entropy is used to penalize the amount of sparsity in
                # feature selection.
                total_entropy = total_entropy + ops.mean(
                    ops.sum(-mask * ops.log(mask + self.epsilon), axis=-1)
                ) / self.num_decision_steps - 1

                masked_features = mask * normalized_inputs

        self.add_loss(total_entropy)

        return decision_out_aggregated

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.decision_step_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'feature_transformer_dim': self.feature_transformer_dim,
            'decision_step_dim': self.decision_step_dim,
            'num_decision_steps': self.num_decision_steps,
            'num_shared_layers': self.num_shared_layers,
            'num_decision_dependent_layers': self.num_decision_dependent_layers,
            'relaxation_factor': self.relaxation_factor,
            'batch_momentum': self.batch_momentum,
            'epsilon': self.epsilon,
            'reset_shared_layers': self.reset_shared_layers,
        })
        return config

    def get_build_config(self):
        build_config = dict(input_shape=self._input_shape)
        return build_config

    def build_from_config(self, build_config):
        self.build(build_config["input_shape"])
