from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layerflow.layers.common import HiLOL
from typing import Union

LAYER_OR_MODEL = Union[keras.layers.Layer, keras.models.Model]


@keras.saving.register_keras_serializable("teras.layerflow.layers")
class Head(HiLOL):
    """
    Head layer to be used on top of a backbone model.

    Args:
        hidden_block: ``layers.Layer`` or ``models.Model``,
            An instance of anything that can serve as the hidden block in the
            classification head.
            It can be as simple as a single ``Dense`` layer, or a custom layer that
            uses a bunch of other dense and other fancy layers,
            or may as well be a keras model -- as long as it satisfies the input
            output constraints.

        output_layer: ``layers.Layer``,
            An instance of keras layer (Dense or a custom layer),
            for classification outputs relevant to the task at hand.
    """
    def __init__(self,
                 hidden_block: LAYER_OR_MODEL,
                 output_layer: layers.Layer,
                 **kwargs):
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)
