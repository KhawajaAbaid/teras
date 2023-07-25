import tensorflow as tf
from teras.layers.numerical_features_extractor import NumericalFeaturesExtractor
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
    return df.values, metadata


def test_numerical_feature_extractor_valid_call(setup_data):
    inputs, metadata = setup_data
    extractor = NumericalFeaturesExtractor(features_metadata=metadata)
    outputs = extractor(inputs)


def test_numerical_feature_extractor_output_shape_when_numerical_features_exist(setup_data):
    inputs, metadata = setup_data
    extractor = NumericalFeaturesExtractor(features_metadata=metadata)
    outputs = extractor(inputs)

    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 1    # there's only one numerical feature


def test_numerical_feature_extractor_raises_error_when_numerical_features_dont_exist():
    df = pd.DataFrame({'player_level': [70, 90, 80, 90, 100, 90, 70, 80],
                       'shirt_number': [10, 7, 7, 10, 10, 10, 7, 10]})
    metadata = get_features_metadata_for_embedding(df,
                                                   categorical_features=['player_level', 'shirt_number'],
                                                   numerical_features=None)
    with pytest.raises(ValueError):
        extractor = NumericalFeaturesExtractor(features_metadata=metadata)
