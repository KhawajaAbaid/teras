from tensorflow import keras
from tensorflow.keras import backend as K
from teras.layerflow.layers.common.common import HiLOL
from teras.utils import (serialize_layers_collection,
                         deserialize_layers_collection)
from typing import List

LIST_OF_LAYERS = List[keras.layers.Layer]


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class SAINTReconstructionHeadBlock(HiLOL):
    """
    SAINTReconstructionHeadBlock layer with LayerFlow design.
    It is used in constructing SAINTReconstructionHead.
    One ``SAINTReconstructionHeadBlock`` is created for each feature in the dataset.

    Args:
        hidden_block: ``keras.layers.Layer``,
            An instance of ``Dense`` layer, or any custom layer that can
            serve as the hidden block.

        output_layer: ``keras.layers.Layer``,
            Any layer that can serve as the output layer BUT it must have
            an output dimensionality equal to the dimensionality of the feature
            it will be applied to. For categorical features, it is equal to the
            number of classes in the feature, and for numerical features,
            it is equal to 1.
    """
    def __init__(self,
                 hidden_block: keras.layers.Layer,
                 output_layer: keras.layers.Layer,
                 **kwargs):
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class SAINTReconstructionHead(keras.layers.Layer):
    """
    SAINTReconstructionHead layer with LayerFlow desing for ``SAINTPretrainer`` model.
    SAINT applies a separate single hidden layer MLP block
    (here we name it, the ReconstructionBlock)
    with an output layer where output dimensions are equal
    to the number of categories in the case of categorical
    features and 1 in the case of numerical features.

    Args:
        reconstruction_blocks: ``List[keras.layers.Layer]``,
            A list of `SAINTRReconstructionHeadBlock` layers - one for each feature,
            where the ``SAINTRReconstructionHeadBlock`` has dimensionality equal to the cardinality
            of that feature.
            For instance, for a categorical feature, the dimensionality of ``SAINTReconstructionBlock``
            will be equal to the number of classes in that feature, while for a numerical feature
            it is just equal to 1.
            You can import the ``SAINTRReconstructionHeadBlock`` layer as follows,
                >>> from teras.layerflow.layers import SAINTReconstructionHeadBlock
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
