import keras

from teras._src.models.tasks.task import Task
from teras._src.api_export import teras_export
from teras._src.typing import ActivationType


@teras_export("teras.models.Regressor")
class Regressor(Task):
    """
    Regressor class that provides a dense prediction head.

    Args:
        backbone: `keras.Model` instance. Backbone is called on
            inputs followed by the dense head that produces predictions.
        num_outputs: int, number of regression outputs to predict.
        hidden_dim: int, hidden dimensionality of the dense head.
            Defaults to 1024.
        hidden_activation: str or callable, activation for the hidden layer.
            Defaults to "relu".
    """
    def __init__(self,
                 backbone: keras.Model,
                 num_outputs: int,
                 hidden_dim: int = 1024,
                 hidden_activation: ActivationType = "relu",
                 **kwargs):
        inputs = backbone.input
        x = backbone(inputs)
        # In case the backbone outputs are of shape (None, a, b),
        # for instance in the case of transformer based models
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(hidden_dim, activation=hidden_activation,
                               name="hidden_layer_regression_head")(x)
        outputs = keras.layers.Dense(num_outputs,
                                     name="predictions")(x)
        super().__init__(inputs, outputs, **kwargs)

        self.backbone = backbone
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "num_outputs": self.num_outputs,
                "hidden_dim": self.hidden_dim,
                "hidden_activation": self.hidden_activation
            }
        )
