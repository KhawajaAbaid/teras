import tensorflow as tf
from teras.layers.saint.saint_numerical_feature_embedding import SAINTNumericalFeatureEmbedding
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import numpy as np


def test_saint_numerical_feature_embedding_output_shape():
    data = pd.DataFrame({"income": np.ones(10),
                         "goals": np.ones(10),
                         'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                         'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
    categorical_feats = ["player_level", "shirt_number"]
    numerical_feats = ["income", "goals"]
    features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                            categorical_features=categorical_feats,
                                                            numerical_features=numerical_feats)
    numerical_embedding = SAINTNumericalFeatureEmbedding(features_metadata=features_metadata,
                                                         embedding_dim=32)
    outputs = numerical_embedding(data.values)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 10    # number of items in each column
    assert tf.shape(outputs)[1] == 2
    assert tf.shape(outputs)[2] == 32
