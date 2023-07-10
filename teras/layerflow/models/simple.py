from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Union


LAYER_OR_MODEL = Union[layers.Layer, keras.Model]


class SimpleModel(models.Model):
    """
    SimpleModel is the simplest teras's LayerFlow model,
    that can be used ot build a model using a body and
    a head.

    Args:
        body: `models.Model | layers.Layer`,
            A layer or model that serves as the body of
            the final model.
        head: `models.Model | layers.Layer`,
            A output layer or a model with an output layer
            that is fit on top of the body layer/model.

    Returns:
        A keras model made up of the body and head layers/models.
    """
    def __init__(self,
                 body: LAYER_OR_MODEL = None,
                 head: LAYER_OR_MODEL = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.body = body
        self.head = head

    def call(self, inputs):
        x = self.body(inputs)
        outputs = self.head(x)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'body': keras.layers.serialize(self.body),
                      'head': keras.layers.serialize(self.head)
                      }
        config.update(new_config)
        return config
