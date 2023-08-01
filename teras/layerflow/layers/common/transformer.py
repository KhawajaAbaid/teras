from tensorflow import keras
from teras.layerflow.layers.common.common import HiLOL
from typing import List

LIST_OF_LAYERS = List[keras.layers.Layer]


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.common")
class FeedForward(HiLOL):
    """
    FeedForward layer with LayerFlow design.
    It is used in the Transformer layer of the ``Encoder`` layer
    of Transformer architecture.
    It is shared by all transformer based architectures for tabular data.

    Args:
        hidden_block: ``keras.layers.Layer``,
            A dense layer or any custom layer that can serve as the hidden
            block for the ``FeedForward`` layer.

        output_layer: ``keras.layers.Layer``,
            A dense layer or any custom layer that projects data back to
            the input/embedding dimensions. Since the ``FeedForward`` receives
            feature embeddings as inputs, it after apply hidden block,
            projects them back to the embedding dimensions.
    """
    def __init__(self,
                 hidden_block: keras.layers.Layer,
                 output_layer: keras.layers.Layer,
                 **kwargs):
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.common")
class Transformer(keras.layers.Layer):
    """
    Transformer layer with LayerFlow design.
    It is used to made up the ``Encoder`` layer of the given
    Transformer-based architecture.

    Args:
        multi_head_attention: ``keras.layers.Layer``,
            An instance of ``MultiHeadAttention`` layer or any layer
            that can be sued in place of ``MultiHeadAttention`` layer.
            You can import this layer as follows,
                >>> from keras.layers import MultiHeadAttention

        feed_forward: ``keras.layers.Layer``,
            An instance of ``FeedForward`` layer or any layer that
            can be used in its place.
            You can import the ``FeedForward`` layer as follows,
                >>> from teras.layerflow.layers import FeedForward

        layer_norm_1: ``keras.layers.Layer``,
            An instance of ``LayerNormalization`` instance applied after
            first skip connection.
            If None, a default ``LayerNormalization`` instance is used.
            You can import this layer as follows,
                >>> from keras.layers import LayerNormalization

        layer_norm_1: ``keras.layers.Layer``, optional,
            An instance of ``LayerNormalization`` instance applied after
            second skip connection.
            If None, a default ``LayerNormalization`` instance is used.
            You can import this layer as follows,
                >>> from keras.layers import LayerNormalization
    """
    def __init__(self,
                 multi_head_attention: keras.layers.Layer,
                 feed_forward: keras.layers.Layer,
                 layer_norm_1: keras.layers.Layer = None,
                 layer_norm_2: keras.layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.multi_head_attention = multi_head_attention
        self.skip_1 = keras.layers.Add(name="skip_1")
        self.layer_norm_1 = layer_norm_1
        if self.layer_norm_1 is None:
            self.layer_norm_1 = keras.layers.LayerNormalization()
        self.feed_forward = feed_forward
        self.skip_2 = keras.layers.Add(name="skip_2")
        self.layer_norm_2 = layer_norm_2
        if self.layer_norm_2 is None:
            self.layer_norm_2 = keras.layers.LayerNormalization()

    def call(self, inputs):
        attention_out = self.multi_head_attention(inputs, inputs)
        x = self.skip_1([attention_out, inputs])
        x = self.layer_norm_1(x)
        feedforward_out = self.feed_forward(x)
        x = self.skip_2([feedforward_out, x])
        x = self.layer_norm_2(x)
        return x

    def get_config(self):
        config = super().get_config()
        new_config = {'multi_head_attention': keras.layers.serialize(self.multi_head_attention),
                      'feed_forward': keras.layers.serialize(self.feed_forward),
                      'layer_norm_1': keras.layers.deserialize(self.layer_norm_1),
                      'layer_norm_2': keras.layers.deserialize(self.layer_norm_2)
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        multi_head_attention = keras.layers.deserialize(config.pop("multi_head_attention"))
        feed_forward = keras.layers.deserialize(config.pop("feed_forward"))
        transformer = keras.layers.deserialize(config.pop("transformer"))
        layer_norm_1 = keras.layers.deserialize(config.pop("layer_norm_1"))
        layer_norm_2 = keras.layers.deserialize(config.pop("layer_norm_2"))
        return cls(multi_head_attention=multi_head_attention,
                   feed_forward=feed_forward,
                   transformer=transformer,
                   layer_norm_1=layer_norm_1,
                   layer_norm_2=layer_norm_2,
                   **config)


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.common")
class Encoder(keras.layers.Layer):
    """
    Encoder layer with LayerFlow design.
    It is made up of several ``Transformer`` layers and encodes the
    input embeddings to useful representations that can then be used
    for the task at hand, such as classificaiton or regression.

    Args:
        transformer_layers: ``List[keras.layers.Layer]``,
            A list of ``Transformer`` layers that make up the encoder
            layer.
            It can also be an instance of Keras Model made up of
            ``Transformer`` or any other layers in the mix.
            You can import the ``Transformer`` layer as follows,
                >>> from teras.layerflow.layers import Transformer
    """
    def __init__(self,
                 transformer_layers: LIST_OF_LAYERS,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer_layers = transformer_layers
        if isinstance(self.transformer_layers, (list, tuple)):
            self.transformer_layers = keras.models.Sequential(
                self.transformer_layers,
                name="transformer_layers"
            )

    def call(self, inputs):
        outputs = self.transformer_layers(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'transformer_layers': keras.layers.serialize(self.transformer_layers)})
        return config

    @classmethod
    def from_config(cls, config):
        transformer_layers = keras.layers.deserialize(config.pop("transformer_layers"))
        return cls(transformer_layers, **config)
