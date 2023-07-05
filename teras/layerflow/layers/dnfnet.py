from tensorflow.keras import layers
from teras.layers.dnfnet import (FeatureSelection,
                                 Localization,
                                 DNNF as _BaseDNNF)
from typing import List

LIST_OF_INT = List[int]


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
