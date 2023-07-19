from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from teras.layerflow.layers.common.common import HiLOL
from teras.utils import (serialize_layers_collection,
                         deserialize_layers_collection)
from typing import List, Union

LIST_OF_LAYERS = List[layers.Layer]
LAYER_OR_MODEL = Union[layers.Layer, models.Model]
LAYERS_COLLECTION = Union[layers.Layer, List[layers.Layer], models.Model]


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class SAINTTransformer(layers.Layer):
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
        multihead_inter_sample_attention: ``layers.Layer``,
            An instance of ``MultiHeadInterSampleAttention`` layer or any other custom
            layer that can work in its place.
            You can import this layer as follows,
                >>> from teras.layers import MultiHeadInterSampleAttention

        feed_forward: ``layers.Layer``,
            An instance of ``FeedForward`` layer or any custom layer that can work
            in its place.
            You can import this layer as follows,
                >>> from teras.layerflow.layers import FeedForward

        transformer: ``layers.Layer``,
            An instance of the regular ``Transformer`` layer, or any custom layer
            that can work in its place.
            You can import this layer as follows,
                >>> from teras.layerflow.layers import Transformer
    """
    def __init__(self,
                 multihead_inter_sample_attention: layers.Layer,
                 feed_forward: layers.Layer,
                 transformer: layers.Layer,
                 **kwargs):
        super().__init__(**kwargs)
        self.multihead_inter_sample_attention = multihead_inter_sample_attention
        self.feed_forward = feed_forward
        self.transformer = transformer

        # We build the inner SAINT Transformer block using keras Functional API
        # Inter Sample Attention Block: this attention is applied to rows.
        inputs = layers.Input(shape=(self._num_features, self._embedding_dim))
        intermediate_outputs = inputs

        if self.apply_attention_to_rows:
            residual = inputs
            x = self.multihead_inter_sample_attention(inputs)
            x = layers.Add()([x, residual])
            x = layers.LayerNormalization()(x)
            residual = x
            x = self.feed_forward(x)
            x = layers.Add()([x, residual])
            intermediate_outputs = layers.LayerNormalization()(x)
            final_outputs = intermediate_outputs

        # MultiHeadAttention block: this attention is applied to columns
        if self.apply_attention_to_features:
            # If `apply_attention_to_features` is set to True,
            # then attention will be applied to columns/features
            # The MultiHeadInterSampleAttention applies attention over rows,
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
        new_config = {'multihead_inter_sample_attention': keras.layers.serialize(self.multihead_inter_sample_attention),
                      'feed_forward': keras.layers.serialize(self.feed_forward),
                      'transformer': keras.layers.serialize(self.transformer),
                      }
        config.update(new_config)
        return config

    @classmethod
    def from_config(cls, config):
        multihead_inter_sample_attention = keras.layers.deserialize(config.pop("multihead_inter_sample_attention"))
        feed_forward = keras.layers.deserialize(config.pop("feed_forward"))
        transformer = keras.layers.deserialize(config.pop("transformer"))
        return cls(multihead_inter_sample_attention=multihead_inter_sample_attention,
                   feed_forward=feed_forward,
                   transformer=transformer,
                   **config)


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class SAINTEncoder(layers.Layer):
    """
    SAINTEncoder layer with LayerFlow desing.
    It is part of the SAINT architecture,
    which is proposed by Gowthami Somepalli et al. in the paper
    SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It simply stacks N transformer layers and applies them to the outputs
    of the embedded features.

    It differs from the typical Encoder block only in that the Transformer
    layer is a bit different from the regular Transformer layer used in the
    Transformer based architectures as it uses multi-head inter-sample attention,
    in addition to the regular mutli-head attention for features.

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        saint_transformer_layers: ``layers.Layer`` or ``List[Layer]`` or ``models.Model``,
            A list, a keras layer or a keras model made up of ``SAINTTransformer`` layers or
            any other custom layers for that purpose.
            By default, 6 ``SAINTTransformer`` layers are used.
            You can import the ``SAINTTransformer`` layer as follows,
                >>> from teras.layerflow.layers.saint import SAINTTransformer
    """
    def __init__(self,
                 saint_transformer_layers: LAYERS_COLLECTION = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.saint_transformer_layers = saint_transformer_layers

        # The reason behind modifying the value of saint_transformer_layers
        # storing that modified value in config and create a new instance using
        # the config of modified value -- instead of just original value
        # is that, this way we can make sure the saved model and reloaded model
        # have the same weights. Otherwise in case of None, if we don't override
        # the value with default layers and store that modified value, instead
        # we just store the original None, when we'll reload the layer,
        # it will re-instantiate the layers -- effectively losing all weights.

        if self.saint_transformer_layers is None:
            # by default, we use 6 SAINTTransformer layers
            self.saint_transformer_layers = keras.models.Sequential(
                [SAINTTransformer() for _ in range(6)],
                name="saint_transformer_layers")

        elif isinstance(self.saint_transformer_layers, (list, tuple)):
            self.saint_transformer_layers = models.Sequential(self.saint_transformer_layers,
                                                              name="saint_transformer_layers")

    def call(self, inputs):
        return self.saint_transformer_layers(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({'saint_transformer_layers': serialize_layers_collection(self.saint_transformer_layers)})
        return config

    @classmethod
    def from_config(cls, config):
        saint_transformer_layers = deserialize_layers_collection(config.pop("saint_transformer_layers"))
        return cls(saint_transformer_layers, **config)


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class ReconstructionBlock(HiLOL):
    """
    ReconstructionBlock layer with LayerFlow design.
    It is used in constructing ReconstructionHead.
    One ``ReconstructionBlock`` is created for each feature in the dataset.

    Args:
        hidden_block: ``layers.Layer``,
            An instance of ``Dense`` layer, or any custom layer that can
            serve as the hidden block.

        output_layer: ``layers.Layer``,
            Any layer that can serve as the output layer BUT it must have
            an output dimensionality equal to the dimensionality of the feature
            it will be applied to. For categorical features, it is equal to the
            number of classes in the feature, and for numerical features,
            it is equal to 1.
    """
    def __init__(self,
                 hidden_block: layers.Layer,
                 output_layer: layers.Layer,
                 **kwargs):
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class ReconstructionHead(layers.Layer):
    """
    ReconstructionHead layer with LayerFlow desing for ``SAINTPretrainer`` model.
    SAINT applies a separate single hidden layer MLP block
    (here we name it, the ReconstructionBlock)
    with an output layer where output dimensions are equal
    to the number of categories in the case of categorical
    features and 1 in the case of numerical features.

    Args:
        reconstruction_blocks: ``List[layers.Layer]``,
            A list of `SAINTReconstructionBlock` layers - one for each feature,
            where the ``SAINTReconstructionBlock`` has dimensionality equal to the cardinality
            of that feature.
            For instance, for a categorical feature, the dimensionality of ``SAINTReconstructionBlock``
            will be equal to the number of classes in that feature, while for a numerical feature
            it is just equal to 1.
            You can import the ``SAINTReconstructionBlock`` layer as follows,
                >>> from teras.layerflow.layers import SAINTReconstructionBlock
    """
    def __init__(self,
                 reconstruction_blocks: LIST_OF_LAYERS,
                 **kwargs):
        super().__init__(**kwargs)
        self.reconstruction_blocks = reconstruction_blocks

    def call(self, inputs):
        """
        Args:
            inputs: SAINT transformer outputs for the augmented data.
                Since we apply categorical and numerical embedding layers
                separately and then combine them into a new features matrix
                this effectively makes the first k features in the outputs
                categorical (since categorical embeddings are applied first)
                and all other features numerical.
                Here, k = num_categorical_features

        Returns:
            Reconstructed input features
        """
        reconstructed_inputs = []
        for idx, block in enumerate(self.reconstruction_blocks):
            feature_encoding = inputs[:, idx]
            reconstructed_feature = block(feature_encoding)
            reconstructed_inputs.append(reconstructed_feature)
        # the reconstructed inputs will have features equal to
        # `number of numerical features` + `number of categories in the categorical features`
        reconstructed_inputs = K.concatenate(reconstructed_inputs, axis=1)
        return reconstructed_inputs

    def get_config(self):
        config = super().get_config()
        config.update({'reconstruction_blocks': serialize_layers_collection(self.reconstruction_blocks),
                       })
        return config

    @classmethod
    def from_config(cls, config):
        reconstruction_blocks = deserialize_layers_collection(config.pop("reconstruction_blocks"))
        return cls(reconstruction_blocks, **config)


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class ProjectionHead(HiLOL):
    """
    ProjectionHead layer with LayerFlow design.
    It is used in the contrastive learning phase of
    the ``SAINTPretrainer`` to project embeddings to a lower dimension.
    According to the SAINT paper,
    "The use of a projection head to reduce dimensionality before computing
    contrastive loss is common in vision and indeed also improves results
    on tabular data."

    Reference(s):
    https://arxiv.org/abs/2106.01342

    Args:
        hidden_block: ``layers.Layer``,
            Hidden block to use in the projection head.
            It can be as simple as a single dense layer with "relu" activation,
            or as complex as you want.
            If the official implementation, the hidden dimensionality is
            computed as below,
            `hidden_dim = 6 * embedding_dim * number_of_features // 5`

        output_layer: ``layers.Layer``,
            Output layer to use in the projection head.
            It should be a simple dense layer that project data to a lower dimension.
            If the official implementation, the output dimensionality is computed as
            below,
            `output_dim = embedding_dim * number_of_features // 5`
    """
    def __init__(self,
                 hidden_block: layers.Layer,
                 output_layer: layers.Layer,
                 **kwargs):
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)
