from tensorflow.keras import layers, models
from teras.layers.tabnet import AttentiveTransformer, FeatureTransformerBlock
from teras.layers.tabnet import (FeatureTransformer as _BaseFeatureTransformer,
                                 Encoder as _BaseEncoder,
                                 ClassificationHead as _BaseClassificationHead,
                                 RegressionHead as _BaseRegressionHead)
from typing import List, Union


LAYERS_COLLECTION = List[layers.Layer]
LAYER_OR_MODEL = Union[layers.Layer, models.Model]


class FeatureTransformer(_BaseFeatureTransformer):
    """
    Feature Transformer with LayerFlow design.
    FeatureTransformer layer is part of the TabNet architecture which
    is proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
        shared_layers: `List[layers.Layer]`,
            A list of FeatureTransformerBlock layers or any custom layer
            that are to be used as the shared layers in the FeatureTransformer.
            You can import the FeatureTransformerBlock as follows:
            >>> from teras.layerflow.layers.tabnet import FeatureTransformerBlock
            and customize the way you want.
            In TabNet architecture, shared layers precede the decision dependent
            layers, and hence if you pass shared_layers, the first layer should
            NOT use the residual batch normalization.
            If you don't want to use any shared layers, you should set
            `num_shared_layers` parameter to 0
        decision_dependent_layers: `List[layers.Layer]`,
            A list of FeatureTransformerBlock layers or any custom layer
            that are to be used as the decision depenedent layers in the FeatureTransformer.
            You can import the FeatureTransformerBlock as follows:
            >>> from teras.layerflow.layers.tabnet import FeatureTransformerBlock
            and customize the way you want.
    """
    def __init__(self,
                 shared_layers: LAYERS_COLLECTION = None,
                 decision_dependent_layers: LAYERS_COLLECTION = None,
                 **kwargs):
        super().__init__(**kwargs)
        if shared_layers is not None:
            shared_layers = self._validate_layers_collection_type(shared_layers,
                                                                  "shared_layers")
            self.shared_layers = shared_layers

        if decision_dependent_layers is not None:
            decision_dependent_layers = self._validate_layers_collection_type(decision_dependent_layers,
                                                                              "decision_dependent_layers")
            self.decision_dependent_layers = decision_dependent_layers

    @staticmethod
    def _validate_layers_collection_type(layers_collection,
                                         parameter_name):
        if isinstance(layers_collection, (list, tuple)):
            return models.Sequential(layers_collection)
        elif isinstance(layers_collection, (layers.Layer, models.Model)):
            return layers_collection
        else:
            raise ValueError(f"Unsupported type of value for `{parameter_name}`. "
                             "Expected type(s): List[layers.Layer], models.Model. "
                             f"Received type: {type(layers_collection)}.")


class Encoder(_BaseEncoder):
    """
    Encoder with LayerFlow design.
    Encoder layer is part of the TabNet architecture which
    is proposed by Sercan et al. in TabNet paper.

    Reference(s):
        https://arxiv.org/abs/1908.07442

    Args:
         feature_transformers_per_step: `List[layers.Layer]`,
            A list of FeatureTransformer layers or any custom layer that can work
            in its place. For each decision step, TabNet uses a separate instance
            of FeatureTransformer layer, hence the number of layers in  the
            `feature_transformers_per_step` determine the total number of decision
            steps to be taken.
        attentive_transformers_per_step: `List[layers.Layer]`,
            A list of AttentiveTransformer layers or any custom layer that can work
            in its place. For each decision step, TabNet uses a separate instance
            of AttentiveTransformer layer, hence the number of layers in  the
            `attentive_transformers_per_step` must be equal to the number of layers
            in the `feature_transformers_per_step` list.
    """
    def __init__(self,
                 feature_transformers_per_step: LAYERS_COLLECTION = None,
                 attentive_transformers_per_step: LAYERS_COLLECTION = None,
                 **kwargs):
        if len(feature_transformers_per_step) != len(attentive_transformers_per_step):
            raise ValueError("Number of layers in the feature_transformers_per_step and attentive_transformers_per_step"
                             " must be equal as for each decision step, there must exist an instance of"
                             "FeatureTransformer and AttentiveTransformer. \n"
                             f"Received, "
                             f"feature_transformers_per_step length: {len(feature_transformers_per_step)}  "
                             f"attentive_transformers_per_step length: {len(attentive_transformers_per_step)}")

        num_decision_steps = len(feature_transformers_per_step)
        super().__init__(num_decision_steps=num_decision_steps,
                         **kwargs)
        if feature_transformers_per_step is not None:
            self.features_transformers_per_step = feature_transformers_per_step

        if attentive_transformers_per_step is not None:
            self.attentive_transformers_per_step = attentive_transformers_per_step


class ClassificationHead(_BaseClassificationHead):
    """
    ClassificationHead with LayerFlow design for TabNet.

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
    RegressionHead with LayerFlow design for TabNet.

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
                 hidden_block: LAYER_OR_MODEL = None,
                 output_layer: layers.Layer = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hidden_block is not None:
            self.hidden_block = hidden_block

        if output_layer is not None:
            self.output_layer = output_layer
