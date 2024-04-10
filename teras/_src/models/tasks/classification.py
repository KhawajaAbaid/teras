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
    """
    def __init__(self,
                 backbone: keras.Model,
                 num_classes: int,
                 activation: ActivationType = "softmax",
                 **kwargs):
        inputs = backbone.input
        x = backbone(inputs)
        # In case the backbone outputs are of shape (None, a, b),
        # for instance in the case of transformer based models
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(num_classes,
                                     activation=activation,
                                     name="predictions")(x)
        super().__init__(inputs, outputs, **kwargs)

        self.backbone = backbone
        self.num_classes = num_classes
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "num_classes": self.num_classes,
                "activation": self.activation
            }
        )
