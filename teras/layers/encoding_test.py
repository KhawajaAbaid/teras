import tensorflow as tf
from teras.layers.encoding import LabelEncoding
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import pytest


@pytest.fixture
def setup_data():
    df = pd.DataFrame({'player_level': [70, 90, 80, 90, 100, 90, 70, 80],
            'shirt_number': [10, 7, 7, 10, 10, 10, 7, 10],
            'income': [1000, 1270, 900, 1000, 1234, 5678, 9999, 0]})
    metadata = get_features_metadata_for_embedding(df,
                                                   categorical_features=["player_level", "shirt_number"],
                                                   numerical_features=["income"])
    return df.values, metadata


def test_label_encoding_output_shape_when_concatenate_numerical_features_is_false(setup_data):
    data, metadata = setup_data
    label_encoding = LabelEncoding(features_metadata=metadata,
                                   concatenate_numerical_features=False)
    outputs = label_encoding(data)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 2    # number of categorical features = 2


def test_label_encoding_output_shape_when_concatenate_numerical_features_is_true(setup_data):
    data, metadata = setup_data
    label_encoding = LabelEncoding(features_metadata=metadata,
                                   concatenate_numerical_features=True)
    outputs = label_encoding(data)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 3    # total number of features = 3

