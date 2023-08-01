from tensorflow import keras
from teras.utils.types import LayersCollection


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class DNFNet(keras.Model):
    """
    DNFNet model based on the DNFNet architecture with LayerFlow design.
    DNFNet architecture is proposed by Liran Katzir et al.
    in the paper,
    "NET-DNF: Effective Deep Modeling Of Tabular Data."

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the input dataset.

        dnnf_layers: ``List[layers.Layer]`` or ``keras.Model``,
            A list of instances of ``DNNF`` layers or
            a Keras layer or model made up of ``DNNF`` layers.
            You can import the ``DNNF`` layer as follows,
                >>> from teras.layers import DNNF

        head: ``keras.layers.Layer``,
            An instance of either ``ClassificationHead`` or ``RegressionHead`` layers,
            depending on the task at hand.
            You can import the ``ClassificationHead`` and ``RegressionHead`` layers as follows,
                >>> from teras.layers import ClassificationHead
                >>> from teras.layers import RegressionHead
    """
    def __init__(self,
                 input_dim: int,
                 dnnf_layers: LayersCollection,
                 head: keras.layers.Layer = None,
                 **kwargs):
        if isinstance(dnnf_layers, (keras.layers.Layer, keras.models.Model)):
            # keep it as is
            dnnf_layers = dnnf_layers
        elif isinstance(dnnf_layers, (list, tuple)):
            dnnf_layers = keras.models.Sequential(dnnf_layers,
                                                  name="dnnf_layers")
        else:
            raise ValueError("`dnnf_layers` can either be a list of `DNNF` layers, a Keras Layer, "
                             f"or a Keras model model but received type: {type(dnnf_layers)}.")
        inputs = keras.layers.Input(shape=(input_dim,))
        outputs = dnnf_layers(inputs)
        if head is not None:
            outputs = head(outputs)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)
        self.input_dim = input_dim
        self.dnnf_layers = dnnf_layers
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'dnnf_layers': keras.layers.serialize(self.dnnf_layers),
                       'head': keras.layers.serialize(self.head)
                       }
                      )
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        dnnf_layers = keras.layers.deserialize(config.pop("dnnf_layers"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(input_dim=input_dim,
                   dnnf_layers=dnnf_layers,
                   head=head,
                   **config)
