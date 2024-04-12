import keras

from teras._src.models.tasks.task import Task
from teras._src.typing import ActivationType
from teras._src.api_export import teras_export


@teras_export("teras.models.Classifier")
class Classifier(Task):
    """
    Classifier class that provides a dense prediction head.

    Args:
        backbone: `keras.Model` instance, backbone is called on
            inputs followed by the dense head that produces predictions.
        num_classes: int, number of classes to predict.
        activation: str or callable, activation function to use for
            outputs. Defaults to "softmax"
        hidden_dim: int, hidden dimensionality of the dense head.
            Defaults to 1024.
        hidden_activation: str or callable, activation for the hidden layer.
            Defaults to "relu".
    """
    def __init__(self,
                 backbone: keras.Model,
                 num_classes: int,
                 activation: ActivationType = "softmax",
                 hidden_dim: int = 1024,
                 hidden_activation: ActivationType = "relu",
                 **kwargs):
        inputs = backbone.input
        x = backbone(inputs)
        # In case the backbone outputs are of shape (None, a, b),
        # for instance in the case of transformer based models
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(hidden_dim, activation=hidden_activation,
                               name="hidden_layer_classification_head")(x)
        outputs = keras.layers.Dense(num_classes,
                                     activation=activation,
                                     name="predictions")(x)
        super().__init__(inputs, outputs, **kwargs)

        self.backbone = backbone
        self.num_classes = num_classes
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "num_classes": self.num_classes,
                "activation": self.activation,
                "hidden_dim": self.hidden_dim,
                "hidden_activation": self.hidden_activation
            }
        )
