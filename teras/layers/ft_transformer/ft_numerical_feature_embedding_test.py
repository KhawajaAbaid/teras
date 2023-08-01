import tensorflow as tf
from teras.layers.ft_transformer.ft_numerical_feature_embedding import FTNumericalFeatureEmbedding
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import pytest


@pytest.fixture
def setup_data():
    df = pd.DataFrame({'player_level': [70, 90, 80, 90, 100, 90, 70, 80],
                       'shirt_number': [10, 7, 7, 10, 10, 10, 7, 10],
                       'income': [1000, 1270, 900, 1000, 1234, 5678, 9999, 0],
                       'goals': [900, 70, 222, 131, 1337, 99, 0, 656]})
    metadata = get_features_metadata_for_embedding(df,
                                                   categorical_features=['player_level', 'shirt_number'],
                                                   numerical_features=['income', 'goals'])
    return df.values, metadata


def test_ft_numerical_feature_embedding_valid_call(setup_data):
    inputs, metadata = setup_data
    normalization = FTNumericalFeatureEmbedding(features_metadata=metadata,
                                                embedding_dim=16)
    outputs = normalization(inputs)


def test_ft_numerical_feature_embedding_output_shape(setup_data):
    inputs, metadata = setup_data
    normalization = FTNumericalFeatureEmbedding(features_metadata=metadata,
                                                embedding_dim=16)
    outputs = normalization(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 8
    # FTNumericalFeatureEmbedding layer returns embedded numerical features only
    assert tf.shape(outputs)[1] == 2
    assert tf.shape(outputs)[2] == 16


def test_ft_numerical_feature_embedding_raises_error_when_numerical_features_dont_exist():
    df = pd.DataFrame({'player_level': [70, 90, 80, 90, 100, 90, 70, 80],
                       'shirt_number': [10, 7, 7, 10, 10, 10, 7, 10]})
    metadata = get_features_metadata_for_embedding(df,
                                                   categorical_features=['player_level', 'shirt_number'],
                                                   numerical_features=None)
    with pytest.raises(ValueError):
        extractor = FTNumericalFeatureEmbedding(features_metadata=metadata)
