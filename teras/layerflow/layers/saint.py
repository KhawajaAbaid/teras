from tensorflow.keras import layers, models
from teras.layers.saint import (NumericalFeatureEmbedding,
                                MultiHeadInterSampleAttention,
                                SAINTTransformer as _BaseSAINTTransformer,
                                Encoder as _BaseEncoder,
                                ReconstructionBlock,
                                ReconstructionHead as _BaseReconstructionHead,
                                ClassificationHead as _BaseClassificationHead,
                                RegressionHead as _BaseRegressionHead)
from typing import List, Union

LIST_OF_LAYERS = List[layers.Layer]
LAYER_OR_MODEL = Union[layers.Layer, models.Model]


class SAINTTransformer(_BaseSAINTTransformer):
    """
    SAINT Transformer layer with LayerFlow design.
    It is part of the SAINT architecture,
    which is proposed by Gowthami Somepalli et al.
    in the paper SAINT: Improved Neural Networks for Tabular Data
    via Row Attention and Contrastive Pre-Training.
    It differs from the usual Transformer (L) block in that it contains additional
    multihead intersample attention layer in addition to the usual multihead attention layer

    Reference(s):
        https://arxiv.org/abs/2106.01342

    Args:
        multihead_inter_sample_attention: `layers.Layer`,
            An instance of `MultiHeadInterSampleAttention` layer or any other custom
            layer that can work in its place.
            You can import this layer as follows,
            >>> from teras.layerflow.layers.saint import MultiHeadInterSampleAttention
        feed_forward: `layers.Layer`,
            An instance of `FeedForward` layer or any custom layer that can work
            in its place.
            You can import this layer as follows,
            >>> from teras.layerflow.layers import FeedForward
        transformer: `layers.Layer`,
            An instance of the regular `Transformer` layer, or any custom layer
            that can work in its place.
            You can import this layer as follows,
            >>> from teras.layerflow.layers import Transformer
    """
    def __init__(self,
                 multihead_inter_sample_attention: layers.Layer = None,
                 feed_forward: layers.Layer = None,
                 transformer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if multihead_inter_sample_attention is not None:
            self.multihead_inter_sample_attention = multihead_inter_sample_attention

        if feed_forward is not None:
            self.feed_forward = feed_forward

        if transformer is not None:
            self.transformer = transformer

        self._build_saint_inner_transformer_block()


class Encoder(_BaseEncoder):
    """
    Encoder for SAINT with LayerFlow desing.
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
        saint_transformer_layers: `List[layers.Layer]`,
             A list of `SAINTTransformer` layers or custom layers
             that can work in their place.
             You can import this layer as follows,
            >>> from teras.layerflow.layers.saint import SAINTTransformer
    """
    def __init__(self,
                 saint_transformer_layers: LIST_OF_LAYERS = None,
                 **kwargs):
        super().__init__(**kwargs)
        if saint_transformer_layers is not None:
            self.transformer_layers = models.Sequential(saint_transformer_layers,
                                                        name="saint_transformer_layers")


class ReconstructionHead(_BaseReconstructionHead):
    """
    ReconstructionHead layer with LayerFlow desing for SAINTPretrainer model.
    SAINT applies a separate single hidden layer MLP block
    (here we name it, the ReconstructionBlock)
    with an output layer where output dimensions are equal
    to the number of categories in the case of categorical
    features and 1 in the case of numerical features.

    Args:
        features_metadata: `dict`,
            a nested dictionary of metadata for features where
            categorical sub-dictionary is a mapping of categorical feature names to a tuple of
            feature indices and the lists of unique values (vocabulary) in them,
            while numerical dictionary is a mapping of numerical feature names to their indices.
            `{feature_name: (feature_idx, vocabulary)}` for feature in categorical features.
            `{feature_name: feature_idx}` for feature in numerical features.
            You can get this dictionary from
                >>> from teras.utils import get_features_metadata_for_embedding
                >>> metadata_dict = get_features_metadata_for_embedding(dataframe,
                                                                        numerical_features,
                                                                        categorical_features)
        reconstruction_blocks: `List[layers.Layer`,
            A list of ReconstructionBlock layers - one for each feature.
             You can import this layer as follows,
            >>> from teras.layerflow.layers.saint import ReconstructionBlock
    """
    def __init__(self,
                 features_metadata: dict,
                 reconstruction_blocks: LIST_OF_LAYERS = None,
                 **kwargs):
        super().__init__(features_metadata=features_metadata,
                         **kwargs)
        if reconstruction_blocks is not None:
            self.reconstruction_blocks = reconstruction_blocks


class ClassificationHead(_BaseClassificationHead):
    """
    ClassificationHead with LayerFlow design for SAINTClassifier.

    Args:
        hidden_block: `layers.Layer | models.Model`,
            An instance of anything that can serve as the hidden block in the
            classification head.
            It can be as simple as a single dense layer, or a custom layer that
            uses a bunch of other dense and other fancy layers,
            or may as well be a keras model -- as long as it satisfies the input
            output constraints.
            If None, a default hidden block specific to the current architecture
            will be used.
        output_layer: `layers.Layer`,
            An instance of keras layer (Dense or a custom layer), with relevant
            activation function for classification relevant to the task at hand.
            If None, a default relevant output layer will be used.
    """
    def __init__(self,
                 hidden_block: LAYER_OR_MODEL = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer


class RegressionHead(_BaseRegressionHead):
    """
    RegressionHead with LayerFlow design for SAINTRegressor.

    Args:
        hidden_block: `layers.Layer | models.Model`,
            An instance of anything that can serve as the hidden block in the
            regression head.
            It can be as simple as a single dense layer, or a custom layer that
            uses a bunch of other dense and other fancy layers,
            or may as well be a keras model -- as long as it satisfies the input
            output constraints.
            If None, a default hidden block specific to the current architecture
            will be used.
        output_layer: `layers.Layer`,
            An instance of keras layer (Dense or a custom layer),
            for regression outputs relevant to the task at hand.
            If None, a default relevant output layer will be used.
    """
    def __init__(self,
                 hidden_block: layers.Layer = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer
