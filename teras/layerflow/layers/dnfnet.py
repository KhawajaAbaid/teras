from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layers.dnfnet import (FeatureSelection,
                                 Localization,
                                 DNNF as _BaseDNNF,
                                 ClassificationHead as _BaseClassificationHead,
                                 RegressionHead as _BaseRegressionHead)
from typing import List, Union

LIST_OF_INT = List[int]
LAYER_OR_MODEL = Union[layers.Layer, models.Model]


class DNNF(_BaseDNNF):
    """
    Disjunctive Normal Neural Form (DNNF) layer with LayeFlow design.
    It is the  main building block of DNF-Net architecture.
    Based on the paper Net-DNF: Effective Deep Modeling of Tabular Data by Liran Katzir et al.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_formulas: `int`, default 256,
            Number of DNF formulas. Each DNF formula is analogous to a tree in tree based ensembles.
        num_conjunctions_arr: `List[int]`, default [6, 9, 12, 15],
            Conjunctions array.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        conjunctions_depth_arr: `List[int]`, default [2, 4, 6],
            Conjunctions depth array.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        feature_selection: `layers.Layer`,
            An instance of `DNFNetFeatureSelection` layer or any custom layer that can work in its place.
            If None, a `DNFNetFeatureSelection` with default values will be used.
            You can import the `DNFNetFeatureSelection` as follows,
                >>> from teras.layerflow.layers import DNFNetFeatureSelection
        localization: `layers.Layer`,
            An instance of `DNFNetLocalization` layer or any custom layer that can work in its place.
            If None, a `DNFNetLocalization` with default values will be used.
            You can import the `DNFNetLocalization` as follows,
                >>> from teras.layerflow.layers import DNFNetLocalization
    """

    def __init__(self,
                 num_formulas: int = 256,
                 num_conjunctions_arr: LIST_OF_INT = [6, 9, 12, 15],
                 conjunctions_depth_arr: LIST_OF_INT = [2, 4, 6],
                 feature_selection: layers.Layer = None,
                 localization: layers.Layer = None,
                 **kwargs):
        super().__init__(num_formulas=num_formulas,
                         num_conjunctions_arr=num_conjunctions_arr,
                         conjunctions_depth_arr=conjunctions_depth_arr,
                         **kwargs)
        if feature_selection is not None:
            self.feature_selection = feature_selection

        if localization is not None:
            self.localization = localization

    def get_config(self):
        config = super().get_config()
        new_config = {'feature_selection': keras.layers.serialize(self.feature_selection),
                      'localization': keras.layers.serialize(self.localization)
                      }
        config.update(new_config)
        return config


class ClassificationHead(_BaseClassificationHead):
    """
    ClassificationHead with LayerFlow design for DNFNet.

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

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config


class RegressionHead(_BaseRegressionHead):
    """
    RegressionHead with LayerFlow design for DNFNet.

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

    def get_config(self):
        config = super().get_config()
        new_config = {'hidden_block': keras.layers.serialize(self.hidden_block),
                      'output_layer': keras.layers.serialize(self.output_layer)
                      }
        config.update(new_config)
        return config
