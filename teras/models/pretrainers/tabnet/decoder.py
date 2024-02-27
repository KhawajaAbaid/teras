import keras
from keras import ops
from teras.layers.tabnet.feature_transformer import TabNetFeatureTransformer
from teras.api_export import teras_export


@teras_export("teras.models.TabNetDecoder")
class TabNetDecoder(keras.Model):
    """
    TabNetDecoder model for self-supervised learning,
    proposed by Arik et al. in the
    "TabNet: Attentive Interpretable Tabular Learning" paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        data_dim: int, The dimensionality of the input dataset.
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
            Defaults to `False`.
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
                 reset_shared_layers: bool = False,
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
        self.reset_shared_layers = reset_shared_layers

        if self.reset_shared_layers:
            TabNetFeatureTransformer.reset_shared_layers()

        self.feature_transformers = [
            TabNetFeatureTransformer(
                hidden_dim=self.feature_transformer_dim,
                num_shared_layers=self.num_decision_steps,
                num_decision_dependent_layers=self.num_decision_dependent_layers,
                batch_momentum=self.batch_momentum,
            )
            for _ in range(self.num_decision_steps)
        ]
        self.projection_layers = [
            keras.layers.Dense(units=self.data_dim,
                               name=f"projection_layer_{i}")
            for i in range(self.num_decision_steps)
        ]

    def call(self, inputs, mask=None):
        batch_size = ops.shape(inputs)[0]
        reconstructed_features = ops.zeros(
            shape=(batch_size, self.data_dim)
        )

        for i in range(self.num_decision_steps):
            feature_out = self.feature_transformers[i](inputs)
            reconstructed_features += self.projection_layers[i](feature_out)

        # According to the paper, the decoder’s last FC (dense) layer is
        # multiplied with S (binary mask indicating which features are
        # missing) to output the unknown features.
        if mask is not None:
            reconstructed_features *= mask

        return reconstructed_features

    def get_config(self):
        config = super().get_config()
        config.update({
            'data_dim': self.data_dim,
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
