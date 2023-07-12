import tensorflow as tf
from teras.layers.embedding import CategoricalFeatureEmbedding
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import pytest


@pytest.fixture
def setup_data():
    data = {'color': ['green', 'yellow', 'green', 'orange', 'red', 'orange', 'orange', 'green'],
            'shape': ['square', 'square', 'circle', 'rectangle', 'circle', 'rectangle', 'rectangle', 'square']}
    metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                   categorical_features=['color', 'shape'])
    categorical_embedding = CategoricalFeatureEmbedding(categorical_features_metadata=metadata["categorical"],
                                                        embedding_dim=32,
                                                        encode=True)
    return data, categorical_embedding


def test_categorical_feature_embedding_accepts_dictionary_data(setup_data):
    categorical_data, categorical_embedding = setup_data
    outputs = categorical_embedding(categorical_data)


def test_categorical_feature_embedding_accepts_array_data(setup_data):
    categorical_data, categorical_embedding = setup_data
    inputs_array = tf.transpose(tf.constant(list(categorical_data.values())))
    outputs = categorical_embedding(inputs_array)


def test_categorical_feature_embedding_output_shape(setup_data):
    categorical_data, categorical_embedding = setup_data
    outputs = categorical_embedding(categorical_data)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 8    # number of items in each column
    assert tf.shape(outputs)[1] == 2
    assert tf.shape(outputs)[2] == 32
