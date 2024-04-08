import keras
from teras._src.api_export import teras_export


@teras_export("teras.layers.LayerList")
class LayerList(keras.layers.Layer):
    """
    LayerList is a list of layers, but is also a layer itself.
    If you know what that means, great. if not, well it's alright.
    you'll get there, one day, sooner or later. just keep pushing. keep
    learning. never give up. you got this, king/queen!

    Args:
        layers: list, list of Keras layers
        sequential: bool, whether to build layers sequentially.
            Set it to True ONLY when each layer in the list is
            applied one after the other in a sequential manner.
            Otherwise, set it to False.
            If sequential, each layer is built using the output
            shape of the previous layer, with first layer being
            built with the `input_shape` argument to the `build` method.
            Otherwise, each layer is built using the `input_shape`
            argument received.
    """
    def __init__(self,
                 layers: list,
                 sequential: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.layers = layers
        self.sequential = sequential

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)

    def compute_output_shape(self, input_shape):
        if not self.sequential:
            return self.layers[-1].compute_output_shape(input_shape)
        else:
            output_shape = input_shape
            for layer in self.layers:
                output_shape = layer.compute_output_shape(output_shape)
            return output_shape

    def build(self, input_shape):
        """
        Build.

        Args:
            input_shape: i wonder what that means
        """
        for layer in self.layers:
            layer.build(input_shape)
            if self.sequential:
                input_shape = layer.compute_output_shape(input_shape)

    def call(self, inputs):
        if self.sequential:
            x = inputs
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            raise NotImplemented(
                "`LayerList` doesn't provide a `call` method for "
                "non-sequential list of layers."
            )

    def get_config(self):
        config = super().get_config()
        serialized_layers = [keras.layers.serialize(layer)
                             for layer in self.layers]
        config.update({
            "layers": serialized_layers,
            "sequential": self.sequential
        })
        return config

    def from_config(cls, config):
        deserialized_layers = [keras.layers.deserialize(layer)
                               for layer in config.pop("layers")]
        return cls(deserialized_layers,
                   **config)
