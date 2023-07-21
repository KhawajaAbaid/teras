from tensorflow import keras
from teras.layerflow.layers.common.common import HiLOL


@keras.saving.register_keras_serializable(package="teras.layerflow.layers.saint")
class SAINTProjectionHead(HiLOL):
    """
    SAINTProjectionHead layer with LayerFlow design.
    It is used in the contrastive learning phase of
    the ``SAINTPretrainer`` to project embeddings to a lower dimension.
    According to the SAINT paper,
    "The use of a projection head to reduce dimensionality before computing
    contrastive loss is common in vision and indeed also improves results
    on tabular data."

    Reference(s):
    https://arxiv.org/abs/2106.01342

    Args:
        hidden_block: ``keras.layers.Layer``,
            Hidden block to use in the projection head.
            It can be as simple as a single dense layer with "relu" activation,
            or as complex as you want.
            If the official implementation, the hidden dimensionality is
            computed as below,
            `hidden_dim = 6 * embedding_dim * number_of_features // 5`

        output_layer: ``keras.layers.Layer``,
            Output layer to use in the projection head.
            It should be a simple dense layer that project data to a lower dimension.
            If the official implementation, the output dimensionality is computed as
            below,
            `output_dim = embedding_dim * number_of_features // 5`
    """
    def __init__(self,
                 hidden_block: keras.layers.Layer,
                 output_layer: keras.layers.Layer,
                 **kwargs):
        super().__init__(hidden_block=hidden_block,
                         output_layer=output_layer,
                         **kwargs)