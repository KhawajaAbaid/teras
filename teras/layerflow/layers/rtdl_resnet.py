from tensorflow.keras import layers, models
from teras.layers.rtdl_resnet import (ResNetBlock,
                                      ClassificationHead as _BaseClassificationHead,
                                      RegressionHead as _BaseRegressionHead)
from typing import Union


LAYER_OR_MODEL = Union[layers.Layer, models.Model]


class ClassificationHead(_BaseClassificationHead):
    """
    ClassificationHead with LayerFlow design for RTDLResNetClassifier.

    Args:
        hidden_block: `layers.Layer | models.Model`,
            An instance of anything that can serve as the hidden block in the
            classification head.
            It can be as simple as a single dense layer, or a custom layer that
            uses a bunch of other dense and other fancy layers,
            or may as well be a keras model -- as long as it satisfies the input
            output constraints.
            If None, a default hidden block specific to the current architecture
            will be used.
        output_layer: `layers.Layer`,
            An instance of keras layer (Dense or a custom layer), with relevant
            activation function for classification relevant to the task at hand.
            If None, a default relevant output layer will be used.
    """
    def __init__(self,
                 hidden_block: LAYER_OR_MODEL = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer


class RegressionHead(_BaseRegressionHead):
    """
    RegressionHead with LayerFlow design for RTDLResNetRegressor.

    Args:
        hidden_block: `layers.Layer | models.Model`,
            An instance of anything that can serve as the hidden block in the
            regression head.
            It can be as simple as a single dense layer, or a custom layer that
            uses a bunch of other dense and other fancy layers,
            or may as well be a keras model -- as long as it satisfies the input
            output constraints.
            If None, a default hidden block specific to the current architecture
            will be used.
        output_layer: `layers.Layer`,
            An instance of keras layer (Dense or a custom layer),
            for regression outputs relevant to the task at hand.
            If None, a default relevant output layer will be used.
    """
    def __init__(self,
                 hidden_block: LAYER_OR_MODEL = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer
