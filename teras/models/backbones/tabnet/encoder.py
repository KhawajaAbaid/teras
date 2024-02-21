import keras
from keras import random, ops
from teras.models.backbones.backbone import Backbone
from teras.layers.tabnet.attentive_transformer import TabNetAttentiveTransformer
from teras.layers.tabnet.feature_transformer import TabNetFeatureTransformer
from teras.api_export import teras_export


@teras_export("teras.models.TabNetEncoderBackbone")
class TabNetEncoderBackbone(Backbone):
    """
    TabNetEncoder proposed by Arik et al. in the
    "TabNet: Attentive Interpretable Tabular Learning" paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: int, The dimensionality of the original input dataset.
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
    """
    def __init__(self,
                 data_dim: int,
                 feature_transformer_dim: int,
                 decision_step_dim: int,
                 num_decision_steps: int = 5,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 5,
                 relaxation_factor: float = 1.5,
                 batch_momentum: float = 0.9,
                 epsilon: float = 1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dim = data_dim
        self.feature_transformer_dim = feature_transformer_dim
        self.decision_step_dim = decision_step_dim
        self.num_decision_steps = num_decision_steps
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon

        self.feature_transformers = [
            TabNetFeatureTransformer(
                hidden_dim=self.feature_transformer_dim,
                num_shared_layers=self.num_decision_steps,
                num_decision_dependent_layers=self.num_decision_dependent_layers,
                batch_momentum=self.batch_momentum,
            )
            for _ in range(self.num_decision_steps)
        ]
        self.attentive_transformers = [
            TabNetAttentiveTransformer(
                data_dim=data_dim,
                batch_momentum=self.batch_momentum
            )
            for _ in range(self.num_decision_steps - 1)
        ]
        self.batch_norm = keras.layers.BatchNormalization(
            momentum=self.batch_momentum
        )

    def call(self, inputs):
        normalized_inputs = self.batch_norm(inputs)
        batch_size = ops.shape(inputs)[0]
        decision_out_aggregated = ops.zeros(
            (batch_size, self.decision_step_dim))
        masked_features = normalized_inputs
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
                prior_scales *= (self.relaxation_factor - mask)

                # Entropy is used to penalize the amount of sparsity in
                # feature selection.
                total_entropy += ops.mean(
                    ops.sum(-mask * ops.log(mask + self.epsilon), axis=-1)
                ) / self.num_decision_steps - 1

                masked_features = mask * normalized_inputs

        self.add_loss(total_entropy)

        return decision_out_aggregated
