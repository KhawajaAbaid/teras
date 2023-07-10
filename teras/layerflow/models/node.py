from tensorflow import keras
from tensorflow.keras import layers, models
from teras.models import NODE as _BaseNODE
from teras.layerflow.layers import NODEClassificationHead, NODERegressionHead
from typing import List
from teras.utils import serialize_layers_collection

LIST_OF_LAYERS = List[layers.Layer]


class NODE(_BaseNODE):
    """
    Neural Oblivious Decision Tree (NODE) model with LayerFlow design.
    It is based on the NODE architecture proposed by Sergei Popov et al.
    in the paper,
    Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        tree_layers: `List[layers.Layer]`,
            A list or tuple of `ObliviousDecisionTree` layers instances, or any custom
            layer that can work in its place.
            If None, default number of `ObliviousDecisionTree` layers with default values
            will be used.
            You can import the `ObliviousDecisionTree` layer as follows,
                >>> from teras.layerflow.layers import ObliviousDecisionTree
        head: `layers.Layer`,
            An instance of `NODEClassificationHead` or `NODERegressionHead`
            layer for final outputs,
            or any layer that can work in place of a head layer for that purpose.
        max_features: `int`, default None,
            Maximum number of input features to use.
            If None, all features in the input dataset will be used.
        input_dropout: `float`, default 0.,
            Dropout rate to apply to inputs.
    """
    def __init__(self,
                 tree_layers: LIST_OF_LAYERS = None,
                 head: layers.Layer = None,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 **kwargs):
        super().__init__(max_features=max_features,
                         input_dropout=input_dropout,
                         **kwargs)
        if tree_layers is not None:
            if not isinstance(tree_layers, (list, tuple)):
                raise ValueError("`tree_layers` must be a list or tuple of `ObliviousDecisionTree` layers, "
                                 f"but recieved type: {type(tree_layers)}.")
            self.tree_layers = tree_layers

        if head is not None:
            self.head = head

    def get_config(self):
        config = super().get_config()
        new_config = {'tree_layers': serialize_layers_collection(self.tree_layers),
                      'head': keras.layers.serialize(self.head)
                      }
        config.update(new_config)
        return config


class NODEClassifier(NODE):
    """
    NODEClassifier with LayerFlow design.
    It is based on the NODE architecture proposed by Sergei Popov et al.
    in the paper,
    Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        tree_layers: `List[layers.Layer]`,
            A list or tuple of `ObliviousDecisionTree` layers instances, or any custom
            layer that can work in its place.
            If None, default number of `ObliviousDecisionTree` layers with default values
            will be used.
            You can import the `ObliviousDecisionTree` layer as follows,
                >>> from teras.layerflow.layers import ObliviousDecisionTree
        head: `layers.Layer`,
            An instance of `NODEClassificationHead` layer for final outputs,
            or any layer that can work in place of a head layer for that purpose.
            You can import the `NODEClassificationHead` as follows,
                >>> from teras.layerflow.layers import NODEClassificationHead
        max_features: `int`, default None,
            Maximum number of input features to use.
            If None, all features in the input dataset will be used.
        input_dropout: `float`, default 0.,
            Dropout rate to apply to inputs.
    """
    def __init__(self,
                 tree_layers: LIST_OF_LAYERS = None,
                 head: layers.Layer = None,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 **kwargs):
        if head is None:
            head = NODEClassificationHead()
        super().__init__(tree_layers=tree_layers,
                         head=head,
                         max_features=max_features,
                         input_dropout=input_dropout,
                         **kwargs)


class NODERegressor(NODE):
    """
    NODERegressor model with LayerFlow design.
    It is based on the NODE architecture proposed by Sergei Popov et al.
    in the paper,
    Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        tree_layers: `List[layers.Layer]`,
            A list or tuple of `ObliviousDecisionTree` layers instances, or any custom
            layer that can work in its place.
            If None, default number of `ObliviousDecisionTree` layers with default values
            will be used.
            You can import the `ObliviousDecisionTree` layer as follows,
                >>> from teras.layerflow.layers import ObliviousDecisionTree
        head: `layers.Layer`,
            An instance of `NODERegressionHead` layer for final outputs,
            or any layer that can work in place of a head layer for that purpose.
            You can import the `NODERegressionHead` as follows,
                >>> from teras.layerflow.layers import NODERegressionHead
        max_features: `int`, default None,
            Maximum number of input features to use.
            If None, all features in the input dataset will be used.
        input_dropout: `float`, default 0.,
            Dropout rate to apply to inputs.
    """
    def __init__(self,
                 tree_layers: LIST_OF_LAYERS = None,
                 head: layers.Layer = None,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 **kwargs):
        if head is None:
            head = NODERegressionHead()
        super().__init__(tree_layers=tree_layers,
                         head=head,
                         max_features=max_features,
                         input_dropout=input_dropout,
                         **kwargs)
