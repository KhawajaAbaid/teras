import keras
from keras import ops
from teras._src.layers.tabnet.feature_transformer_layer import TabNetFeatureTransformerLayer
from teras._src.api_export import teras_export
from teras._src.layers.layer_list import LayerList


@teras_export("teras.layers.TabNetFeatureTransformer")
class TabNetFeatureTransformer(keras.layers.Layer):
    """
    FeatureTransformer layer based on the TabNet architecture proposed
    in the "TabNet: Attentive Interpretable Tabular Learning" paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        hidden_dim: int, hidden dimensionality of the feature transformer.
        num_shared_layers: int, number of shared layers to use.
            Defaults to 2.
        num_decision_dependent_layers: int, number of decision dependent
            layers to use. Defaults to 2.
        batch_momentum: float, batch momentum
    """
    shared_layers = None

    def __init__(self,
                 hidden_dim: int,
                 num_shared_layers: int = 2,
                 num_decision_dependent_layers: int = 2,
                 batch_momentum: float = 0.99,
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_shared_layers = num_shared_layers
        self.num_decision_dependent_layers = num_decision_dependent_layers
        self.batch_momentum = batch_momentum

        self._maybe_create_shared_layers(self.num_shared_layers,
                                         self.hidden_dim,
                                         self.batch_momentum)
        self.decision_dependent_layers = LayerList([
            TabNetFeatureTransformerLayer(
                dim=hidden_dim,
                batch_momentum=batch_momentum,
                name=f"decision_dependent_layer_{i+1}")
            for i in range(num_decision_dependent_layers)
        ],
            name="decision_dependent_layers"
        )

    @classmethod
    def _maybe_create_shared_layers(cls, num_layers, hidden_dim,
                                    batch_momentum):
        if cls.shared_layers:
            return
        cls.shared_layers = LayerList([TabNetFeatureTransformerLayer(
            dim=hidden_dim,
            batch_momentum=batch_momentum,
            name=f"shared_layer_{i+1}"
        ) for i in range(num_layers)],
            name="shared_layers"
        )

    @classmethod
    def reset_shared_layers(cls):
        cls.shared_layers = None

    def build(self, input_shape):
        # Shared layers need only be built once, because they are
        # shared across instances of `TabNetFeatureTransformer`
        if not self.shared_layers.built:
            self.shared_layers.build(input_shape)
        input_shape = self.shared_layers.compute_output_shape(input_shape)
        self.decision_dependent_layers.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = inputs
        residue = None
        for layer in self.shared_layers:
            x = layer(x)
            if residue is not None:
                x += ops.sqrt(0.5) * residue
            residue = x

        for layer in self.decision_dependent_layers:
            x = layer(x)
            if residue is not None:
                x += ops.sqrt(0.5) * residue
            residue = x
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_shared_layers": self.num_shared_layers,
            "num_decision_dependent_layers":
                self.decision_dependent_layers,
            "batch_momentum": self.batch_momentum,
        })
