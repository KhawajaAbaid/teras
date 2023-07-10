from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.ft_transformer import (NumericalFeatureEmbedding,
                                         CLSToken,
                                         ClassificationHead as _BaseClassificationHead,
                                         RegressionHead as _BaseRegressionHead)


class ClassificationHead(_BaseClassificationHead):
    """
    ClassificationHead with LayerFlow design for FTTransformer.

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
                 hidden_block: layers.Layer = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config


class RegressionHead(_BaseRegressionHead):
    """
    RegressionHead with LayerFlow design for FTTransformer.

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
                 hidden_block: layers.Layer = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config
