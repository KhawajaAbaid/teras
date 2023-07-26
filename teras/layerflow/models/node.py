from tensorflow import keras
from typing import List
from teras.utils.utils import (serialize_layers_collection,
                               deserialize_layers_collection)
from teras.layers.node.node_feature_selector import NodeFeatureSelector

LIST_OF_LAYERS = List[keras.layers.Layer]


@keras.saving.register_keras_serializable(package="teras.layerflow.models")
class NODE(keras.Model):
    """
    Neural Oblivious Decision Tree (NODE) model with LayerFlow design.
    It is based on the NODE architecture proposed by Sergei Popov et al.
    in the paper,
    Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or, the number of features in the dataset.

        tree_layers: ``List[layers.Layer]``,
            A list or tuple of `ObliviousDecisionTree` layers instances, or any custom
            layer that can work in its place.
            If None, default number of `ObliviousDecisionTree` layers with default values
            will be used.
            You can import the `ObliviousDecisionTree` layer as follows,
                >>> from teras.layerflow.layers import ObliviousDecisionTree

        feature_selector: ``keras.layers.Layer``:
            An instance of ``NodeFeatureSelector`` layer that selects features based on
            In None, all features will be used (no ``NodeFeatureSelector`` layer will be applied).
            You can import the ``NodeFeatureSelector`` layer as follows,
                >>> from teras.layers import NodeFeatureSelector

        dropout: `keras.layers.Layer``,
            An instance of ``Dropout`` layer to apply over inputs.
            If None, no dropout will be applied.
            You can import the ``Dropout`` layer as follows,
                >>> from keras.layers import Dropout

        head: ``keras.layers.Layer``,
            An instance of ``Head`` layer to make classification or regression predictions.
            In case you're using this model as a base model for pretraining, you MUST leave
            this argument as None.
            You can import the ``Head`` layer as follows,
                >>> from teras.layerflow.layers import Head
            or you can just use your own custom layer.
    """
    def __init__(self,
                 input_dim: int,
                 tree_layers: LIST_OF_LAYERS = None,
                 feature_selector: keras.layers.Layer = None,
                 dropout: keras.layers.Layer = None,
                 head: keras.layers.Layer = None,
                 **kwargs):
        if not isinstance(tree_layers, (list, tuple)):
            raise ValueError("`tree_layers` must be a list or tuple of `ObliviousDecisionTree` layers, "
                             f"but received type: {type(tree_layers)}.")

        inputs = keras.layers.Input(shape=(input_dim,),
                                    name="inputs")
        x_out = inputs
        initial_features = input_dim
        for layer in tree_layers:
            x = x_out
            if feature_selector is not None:
                x = feature_selector(x)
            if dropout is not None:
                x = dropout(x)
            h = layer(x)
            x_out = keras.layers.Concatenate(axis=-1)([x_out, h])
        outputs = x_out[..., initial_features:]
        if head is not None:
            outputs = head(outputs)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)

        self.tree_layers = tree_layers
        self.head = head

    def get_config(self):
        config = super().get_config()
        config.update({'tree_layers': serialize_layers_collection(self.tree_layers),
                       'head': keras.layers.serialize(self.head),
                       'max_features': self.max_features,
                       'input_dropout': self.input_dropout
                       })
        return config

    @classmethod
    def from_config(cls, config):
        tree_layers = deserialize_layers_collection(config.pop("tree_layers"))
        head = keras.layers.deserialize(config.pop("head"))
        return cls(tree_layers=tree_layers,
                   head=head,
                   **config)
