import tensorflow as tf
from teras.layers.encoding import LabelEncoding
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import pytest


@pytest.fixture
def setup_data():
    data = {'color': ['green', 'yellow', 'green', 'orange', 'red', 'orange', 'orange', 'green'],
            'shape': ['square', 'square', 'circle', 'rectangle', 'circle', 'rectangle', 'rectangle', 'square'],
            'area': [8., 10., 11., 7., 5., 1., 7., 9.]}
    metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                   categorical_features=['color', 'shape'],
                                                   numerical_features=['area'])
    return data, metadata


def test_label_encoding_output_shape_when_concatenate_numerical_features_is_false(setup_data):
    data, metadata = setup_data
    label_encoding = LabelEncoding(categorical_features_metadata=metadata["categorical"],
                                   concatenate_numerical_features=False)
    outputs = label_encoding(data)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 2    # number of categorical features = 2


def test_label_encoding_output_shape_when_concatenate_numerical_features_is_true(setup_data):
    data, metadata = setup_data
    label_encoding = LabelEncoding(categorical_features_metadata=metadata["categorical"],
                                   concatenate_numerical_features=True)
    outputs = label_encoding(data)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 8
    assert tf.shape(outputs)[1] == 3    # total number of features = 3

