import tensorflow as tf
from teras.layers.ft_transformer import NumericalFeatureEmbedding
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import pytest


@pytest.fixture
def setup_data():
    data = {'length': tf.ones(shape=(10,)),
            'area': tf.ones(shape=(10,))}
    metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                   numerical_features=['length', 'area'])
    numerical_embedding = NumericalFeatureEmbedding(numerical_features_metadata=metadata["numerical"],
                                                    embedding_dim=32)
    return data, numerical_embedding


def test_ft_numerical_feature_embedding_accepts_dictionary_data(setup_data):
    numerical_data, numerical_embedding = setup_data
    outputs = numerical_embedding(numerical_data)


def test_ft_numerical_feature_embedding_accepts_array_data(setup_data):
    numerical_data, numerical_embedding = setup_data
    inputs_array = tf.transpose(list(numerical_data.values()))
    outputs = numerical_embedding(inputs_array)


def test_ft_numerical_feature_embedding_output_shape(setup_data):
    numerical_data, numerical_embedding = setup_data
    outputs = numerical_embedding(numerical_data)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 10    # number of items in each column
    assert tf.shape(outputs)[1] == 2
    assert tf.shape(outputs)[2] == 32
