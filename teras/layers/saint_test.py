import tensorflow as tf
from teras.layers.saint import (NumericalFeatureEmbedding,
                                MultiHeadInterSampleAttention,
                                SAINTTransformer,
                                Encoder,
                                ProjectionHead,
                                ReconstructionBlock,
                                ReconstructionHead)
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import pytest


@pytest.fixture
def numerical_embedding_setup_data():
    data = {'length': tf.ones(shape=(10,)),
            'area': tf.ones(shape=(10,))}
    metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                   numerical_features=['length', 'area'])
    numerical_embedding = NumericalFeatureEmbedding(numerical_features_metadata=metadata["numerical"],
                                                    embedding_dim=32)
    return data, numerical_embedding


def test_saint_numerical_feature_embedding_accepts_dictionary_data(numerical_embedding_setup_data):
    numerical_data, numerical_embedding = numerical_embedding_setup_data
    outputs = numerical_embedding(numerical_data)


def test_saint_numerical_feature_embedding_accepts_array_data(numerical_embedding_setup_data):
    numerical_data, numerical_embedding = numerical_embedding_setup_data
    inputs_array = tf.transpose(list(numerical_data.values()))
    outputs = numerical_embedding(inputs_array)


def test_saint_numerical_feature_embedding_output_shape(numerical_embedding_setup_data):
    numerical_data, numerical_embedding = numerical_embedding_setup_data
    outputs = numerical_embedding(numerical_data)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 10    # number of items in each column
    assert tf.shape(outputs)[1] == 2
    assert tf.shape(outputs)[2] == 32


def test_saint_multi_head_inter_sample_attention_output_shape():
    multihead_attention = MultiHeadInterSampleAttention(key_dim=32)
    inputs = tf.ones(shape=(128, 16, 32))
    outputs = multihead_attention(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 128   # number of items in each column
    assert tf.shape(outputs)[1] == 16
    assert tf.shape(outputs)[2] == 32


def test_saint_transformer_output_shape():
    saint_transformer = SAINTTransformer(embedding_dim=32,
                                         num_embedded_features=16)
    inputs = tf.ones((128, 16, 32), dtype=tf.float32)
    outputs = saint_transformer(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16
    assert tf.shape(outputs)[2] == 32


def test_encoder_output_shape():
    encoder = Encoder(embedding_dim=32,
                      num_embedded_features=16)
    inputs = tf.ones((128, 16, 32), dtype=tf.float32)
    outputs = encoder(inputs)
    assert len(tf.shape(outputs)) == 3
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 16
    assert tf.shape(outputs)[2] == 32
