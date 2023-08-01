import tensorflow as tf
from teras.layers.saint.saint_reconstruction_head import SAINTReconstructionHead
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import numpy as np


def test_saint_reconstruction_head_output_shape():
    data = pd.DataFrame({"income": np.ones(10),
                         "goals": np.ones(10),
                         'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                         'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
    categorical_feats = ["player_level", "shirt_number"]
    numerical_feats = ["income", "goals"]
    features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                            categorical_features=categorical_feats,
                                                            numerical_features=numerical_feats)
    sum_of_features_cardinalities = sum(map(lambda x: len(features_metadata["categorical"][x][1]),
                                            features_metadata["categorical"])
                                        ) + len(numerical_feats)

    saint_reconstruction_head = SAINTReconstructionHead(features_metadata,
                                                        embedding_dim=32)
    inputs = tf.ones(shape=(16, 4, 32), dtype=tf.float32)
    outputs = saint_reconstruction_head(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 16
    assert tf.shape(outputs)[1] == sum_of_features_cardinalities
