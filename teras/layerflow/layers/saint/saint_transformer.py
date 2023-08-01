from tensorflow import keras


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class SAINTTransformer(keras.layers.Layer):
    """
    SAINT Transformer layer with LayerFlow design.
    It is part of the SAINT architecture,
    which is proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It differs from the usual Transformer (L) block in that it contains additional
    ``MultiHeadInterSampleAttention`` layer in addition to the usual
    ``MultiHeadAttention`` layer.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        multi_head_inter_sample_attention: ``keras.layers.Layer``,
            An instance of ``MultiHeadInterSampleAttention`` layer or any other custom
            layer that can work in its place.
            You can import this layer as follows,
                >>> from teras.layers import MultiHeadInterSampleAttention

        feed_forward: ``keras.layers.Layer``,
            An instance of ``FeedForward`` layer or any custom layer that can work
            in its place.
            You can import this layer as follows,
                >>> from teras.layerflow.layers import FeedForward

        transformer: ``keras.layers.Layer``,
            An instance of the regular ``Transformer`` layer, or any custom layer
            that can work in its place.
            You can import this layer as follows,
                >>> from teras.layerflow.layers import Transformer
    """
    def __init__(self,
                 multi_head_inter_sample_attention: keras.layers.Layer,
                 feed_forward: keras.layers.Layer,
                 transformer: keras.layers.Layer,
                 **kwargs):
        super().__init__(**kwargs)
        self.multi_head_inter_sample_attention = multi_head_inter_sample_attention
        self.feed_forward = feed_forward
        self.transformer = transformer

    def build(self, input_shape):
        # We build the inner SAINT Transformer block using keras Functional API
        # and since we need the input dimensions that's why we're building it in the build method.

        # Inter Sample Attention Block: this attention is applied to rows.
        inputs = keras.layers.Input(shape=tuple(input_shape[1:]))
        intermediate_outputs = inputs
        if self.apply_attention_to_rows:
            residual = inputs
            x = self.multi_head_inter_sample_attention(inputs)
            x = keras.layers.Add()([x, residual])
            x = keras.layers.LayerNormalization()(x)
            residual = x
            x = self.feed_forward(x)
            x = keras.layers.Add()([x, residual])
            intermediate_outputs = keras.layers.LayerNormalization()(x)
            final_outputs = intermediate_outputs

        # MultiHeadAttention block: this attention is applied to columns
        if self.apply_attention_to_features:
            # If `apply_attention_to_features` is set to True,
            # then attention will be applied to columns/features
            # The MultiHeadInterSampleAttention appollection)lies attention over rows,
            # but the regular MultiHeadAttention layer is used to apply attention over features.
            # Since the common Transformer layer applies MutliHeadAttention over features
            # as well as takes care of applying all the preceding and following layers,
            # so we'll just use that here.
            final_outputs = self.transformer(intermediate_outputs)

        self.transformer_block = keras.Model(inputs=inputs,
                                             outputs=final_outputs,
                                             name="saint_inner_transformer_block")

    def call(self, inputs):
        outputs = self.transformer_block(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'multi_head_inter_sample_attention': keras.layers.serialize(self.multi_head_inter_sample_attention),
                      'feed_forward': keras.layers.serialize(self.feed_forward),
                      'transformer': keras.layers.serialize(self.transformer),
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        multi_head_inter_sample_attention = keras.layers.deserialize(config.pop("multi_head_inter_sample_attention"))
        feed_forward = keras.layers.deserialize(config.pop("feed_forward"))
        transformer = keras.layers.deserialize(config.pop("transformer"))
        return cls(multi_head_inter_sample_attention=multi_head_inter_sample_attention,
                   feed_forward=feed_forward,
                   transformer=transformer,
                   **config)
