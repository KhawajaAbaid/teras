import tensorflow as tf
from teras.layers.categorical_feature_embedding import CategoricalFeatureEmbedding
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import pytest


@pytest.fixture
def setup_data():
    df = pd.DataFrame({'player_level': [70, 90, 80, 90, 100, 90, 70, 80],
                       'shirt_number': [10, 7, 7, 10, 10, 10, 7, 10],
                       'income': [1000, 1270, 900, 1000, 1234, 5678, 9999, 0]})
    metadata = get_features_metadata_for_embedding(df,
                                                   categorical_features=['player_level', 'shirt_number'],
                                                   numerical_features=['income'])
    categorical_embedding = CategoricalFeatureEmbedding(features_metadata=metadata,
                                                        embedding_dim=32,
                                                        encode=True)
    return df.values, categorical_embedding


def test_categorical_feature_embedding_valid_call(setup_data):
    data, categorical_embedding = setup_data
    outputs = categorical_embedding(data)


def test_categorical_feature_embedding_output_shape(setup_data):
    data, categorical_embedding = setup_data
    outputs = categorical_embedding(data)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 8    # number of items in each column
    assert tf.shape(outputs)[1] == 2
    assert tf.shape(outputs)[2] == 32


def test_categorical_feature_embedding_raises_error_when_categorical_features_dont_exist():
    df = pd.DataFrame({'income': [1000, 1270, 900, 1000, 1234, 5678, 9999, 0],
                       'goals': [900, 70, 222, 131, 1337, 99, 0, 656]})
    metadata = get_features_metadata_for_embedding(df,
                                                   categorical_features=None,
                                                   numerical_features=['income', 'goals'])
    with pytest.raises(ValueError):
        extractor = CategoricalFeatureEmbedding(features_metadata=metadata)
