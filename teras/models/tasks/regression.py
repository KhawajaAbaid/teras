import keras
from teras.models.tasks.task import Task


class Regressor(Task):
    """
    Regressor class that provides a dense prediction head.

    Args:
        backbone: `keras.Model` instance. Backbone is called on
            inputs followed by the dense head that produces predictions.
        num_outputs: int, number of regression outputs to predict.
    """
    def __init__(self,
                 backbone: keras.Model,
                 num_outputs: int,
                 **kwargs):
        inputs = backbone.input
        x = backbone(inputs)
        # In case the backbone outputs are of shape (None, a, b),
        # for instance in the case of transformer based models
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(num_outputs,
                                     name="predictions")(x)
        super().__init__(inputs, outputs, **kwargs)

        self.backbone = backbone
        self.num_outputs = num_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.layers.serialize(self.backbone),
                "num_outputs": self.num_outputs
            }
        )
