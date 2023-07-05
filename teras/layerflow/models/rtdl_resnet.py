from tensorflow.keras import layers, models
from teras.models.rtdl_resnet import RTDLResNet as _BaseRTDLResNet
from teras.layerflow.layers import RTDLResNetClassificationHead, RTDLResNetRegressionHead
from typing import List, Union


PACK_OF_LAYERS = Union[List[layers.Layer], models.Model]


class RTDLResNet(_BaseRTDLResNet):
    """
    RTDLResNet model with LayerFlow desing.
    It is based on the ResNet architecture proposed by Yury Gorishniy et al.
    in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        resnet_blocks: `List[layers.Layer] | models.Model`,
            List of `RTDLResNetBlock` layers to use in the RTDLResNet model.
            You can pass either pass a list of instances of `RTDLResNetBlock` layers,
            or pack them in a Keras model.
            You can import the `RTDLResNetBlock` layer as follows,
                >>> from teras.layerflow.layers import RTDLResNetBlock
        head: `layers.Layer`,
            An instance of `RTDLResNetClassificationHead` or `RTDLResNetRegressionHead`
            layer for final outputs,
            or any layer that can work in place of a head layer for that purpose.
    """

    def __init__(self,
                 resnet_blocks: PACK_OF_LAYERS = None,
                 head: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if resnet_blocks is not None:
            if isinstance(resnet_blocks, models.Model):
                # Leave it as it
                resnet_blocks = resnet_blocks
            elif isinstance(resnet_blocks, (list, tuple)):
                resnet_blocks = models.Sequential(resnet_blocks,
                                                  name="resnet_blocks")
            else:
                raise ValueError("`resnet_blocks` can either be a list of `RTDLResNetBlock` layers "
                                 f"or a keras model but received type: {type(resnet_blocks)}.")
            self.resnet_blocks = resnet_blocks

        if head is not None:
            self.head = head


class RTDLResNetClassifier(RTDLResNet):
    """
    RTDLResNetClassifier with LayerFlow design.
    It is based on the ResNet architecture proposed by Yury Gorishniy et al.
    in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        resnet_blocks: `List[layers.Layer] | models.Model`,
            List of `RTDLResNetBlock` layers to use in the RTDLResNet model.
            You can pass either pass a list of instances of `RTDLResNetBlock` layers,
            or pack them in a Keras model.
            You can import the `RTDLResNetBlock` layer as follows,
                >>> from teras.layerflow.layers import RTDLResNetBlock
        head: `layers.Layer`,
            An instance of `RTDLResNetClassificationHead`  layer for final outputs,
            or any layer that can work in place of this layer for the purpose.
            You can import the `RTDLResNetClassificationHead` layer as follows,
                >>> from teras.layerflow.layers import RTDLResNetClassificationHead
    """
    def __init__(self,
                 resnet_blocks: PACK_OF_LAYERS = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = RTDLResNetClassificationHead()
        super().__init__(resnet_blocks=resnet_blocks,
                         head=head,
                         **kwargs)


class RTDLResNetRegressor(RTDLResNet):
    """
    RTDLResNetRegressor with LayerFlow design.
    It is based on the ResNet architecture proposed by Yury Gorishniy et al.
    in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        resnet_blocks: `List[layers.Layer] | models.Model`,
            List of `RTDLResNetBlock` layers to use in the RTDLResNet model.
            You can pass either pass a list of instances of `RTDLResNetBlock` layers,
            or pack them in a Keras model.
            You can import the `RTDLResNetBlock` layer as follows,
                >>> from teras.layerflow.layers import RTDLResNetBlock
        head: `layers.Layer`,
            An instance of `RTDLResNetRegressionHead`  layer for final outputs,
            or any layer that can work in place of this layer for the purpose.
            You can import the `RTDLResNetRegressionHead` layer as follows,
                >>> from teras.layerflow.layers import RTDLResNetRegressionHead
    """
    def __init__(self,
                 resnet_blocks: PACK_OF_LAYERS = None,
                 head: layers.Layer = None,
                 **kwargs):
        if head is None:
            head = RTDLResNetRegressionHead()
        super().__init__(resnet_blocks=resnet_blocks,
                         head=head,
                         **kwargs)
