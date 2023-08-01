from tensorflow import keras
from teras.utils.types import LayersCollection
from typeguard import check_type


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class RTDLResNet(keras.Model):
    """
    RTDLResNet model with LayerFlow desing.
    It is based on the ResNet architecture proposed by Yury Gorishniy et al.
    in the paper,
    Revisiting Deep Learning Models for Tabular Data.

    Reference(s):
        https://arxiv.org/abs/2106.11959

    Args:
        input_dim: ``int``,
            The dimensionality of the input dataset,
            or the number of features in the input dataset.

        resnet_blocks: ``List[layers.Layer]`` or ``models.Model`` or ``keras.layers.Layer``,
            List of ``RTDLResNetBlock`` layers to use in the ``RTDLResNet`` model.
            You can pass either pass a list of instances of ``RTDLResNetBlock`` layers,
            or pack them in a Keras model.
            You can import the ``RTDLResNetBlock`` layer as follows,
                >>> from teras.layers import RTDLResNetBlock

        head: ``keras.layers.Layer``,
            An instance of either ``ClassificationHead`` or ``RegressionHead`` layers,
            depending on the task at hand.

            You can import the ``ClassificationHead`` and ``RegressionHead`` layers as follows,
                >>> from teras.layers import ClassificationHead
                >>> from teras.layers import RegressionHead
    """

    def __init__(self,
                 input_dim: int,
                 resnet_blocks: LayersCollection,
                 head: keras.layers.Layer = None,
                 **kwargs):
        # if not isinstance(resnet_blocks, LayersCollection):
        try:
            check_type("resnet_blocks", resnet_blocks, LayersCollection)
        except TypeError:
            raise TypeError("`resnet_blocks` can either be a list of `RTDLResNetBlock` layers "
                            "or a Keras Layer or Model made up of `RTDLResNetBlock` layers. \n"
                            f"But received type: {type(resnet_blocks)}.")
        if isinstance(resnet_blocks, (list, tuple)):
            resnet_blocks = keras.models.Sequential(resnet_blocks,
                                                    name="resnet_blocks")

        inputs = keras.layers.Input(shape=(input_dim,))
        outputs = resnet_blocks(inputs)
        if head is not None:
            outputs = head(outputs)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)
        self.input_dim = input_dim
        self.resnet_blocks = resnet_blocks
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'resnet_blocks': keras.layers.serialize(self.resnet_blocks),
                       'head': keras.layers.serialize(self.head)
                       })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        resnet_blocks = keras.layers.deserialize(config.pop("resnet_blocks"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(input_dim=input_dim,
                   resnet_blocks=resnet_blocks,
                   head=head,
                   **config)
