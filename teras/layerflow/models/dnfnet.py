from tensorflow import keras
from tensorflow.keras import layers, models
from teras.layerflow.layers import DNNF, DNFNetClassificationHead, DNFNetRegressionHead
from teras.models import (DNFNet as _BaseDNFNet)
from typing import List, Union
from teras.utils import serialize_layers_collection

LIST_OF_INT = List[int]
LIST_OF_FLOAT = List[float]
PACK_OF_LAYERS = Union[layers.Layer, List[layers.Layer], models.Model]


class DNFNet(_BaseDNFNet):
    """
    DNFNet model based on the DNFNet architecture with LayerFlow design.
    DNFNet architecture is proposed by Liran Katzir et al.
    in the paper,
    "NET-DNF: Effective Deep Modeling Of Tabular Data."

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        dnnf_layers: `List[layers.Layer] | keras.Model`,
            List of `DNNF` layers to use in the DNFNet model.
            You can pass either pass a list of instances of `DNNF` layers,
            or pack them in a Keras model.
            You can import the DNNF layer as follows,
                >>> from teras.layerflow.layers import DNNF
        head: `layers.Layer`,
            An instance of `DNFNetClassificationHead` or `DNFNetRegressionHead`
            layer for final outputs,
            or any layer that can work in place of a head layer for that purpose.
    """
    def __init__(self,
                 dnnf_layers: PACK_OF_LAYERS = None,
                 head: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if dnnf_layers is not None:
            if isinstance(dnnf_layers, (layers.Layer, models.Model)):
                # keep it as is
                dnnf_layers = dnnf_layers
            elif isinstance(dnnf_layers, (list, tuple)):
                dnnf_layers = models.Sequential(dnnf_layers,
                                                name="dnnf_layers")
            else:
                raise ValueError("`dnnf_layers` can either be a Keras Layer, list of `DNNF` layers "
                                 f"or a Keras model model but received type: {type(dnnf_layers)}.")
            self.dnnf_layers = dnnf_layers

        if head is not None:
            self.head = head

    def get_config(self):
        config = super().get_config()
        new_config = {'dnnf_layers': serialize_layers_collection(self.dnnf_layers),
                      'head': keras.layers.serialize(self.head)
                      }
        config.update(new_config)
        return config


class DNFNetClassifier(DNFNet):
    """
    DNFNetClassifier with LayerFlow design.
    It is based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper,
    NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        dnnf_layers: `List[layers.Layer] | keras.Model`,
            List of `DNNF` layers to use in the DNFNet model.
            You can pass either pass a list of instances of `DNNF` layers,
            or pack them in a Keras model.
            You can import the DNNF layer as follows,
                >>> from teras.layerflow.layers import DNNF
        head: `layers.Layer`,
            An instance of `DNFNetClassificationHead`  layer for final outputs,
            or any layer that can work in place of this layer for the purpose.
            You can import the `DNFNetClassificationHead` layer as follows,
                >>> from teras.layerflow.layers import DNFNetClassificationHead
    """
    def __init__(self,
                 dnnf_layers: PACK_OF_LAYERS = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = DNFNetClassificationHead()
        super().__init__(dnnf_layers=dnnf_layers,
                         head=head,
                         **kwargs)


class DNFNetRegressor(DNFNet):
    """
    DNFNetRegressor based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper,
    NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        dnnf_layers: `List[layers.Layer] | keras.Model`,
            List of `DNNF` layers to use in the DNFNet model.
            You can pass either pass a list of instances of `DNNF` layers,
            or pack them in a Keras model.
            You can import the DNNF layer as follows,
                >>> from teras.layerflow.layers import DNNF
        head: `layers.Layer`,
            An instance of `DNFNetRegressionHead`  layer for final outputs,
            or any layer that can work in place of this layer for the purpose.
            You can import the `DNFNetRegressionHead` layer as follows,
                >>> from teras.layerflow.layers import DNFNetRegressionHead
    """
    def __init__(self,
                 dnnf_layers: PACK_OF_LAYERS = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = DNFNetRegressionHead()
        super().__init__(dnnf_layers=dnnf_layers,
                         head=head,
                         **kwargs)
