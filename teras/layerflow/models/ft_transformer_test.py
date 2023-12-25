from teras.layerflow.models import FTTransformer
from teras.layers.ft_transformer import FTNumericalFeatureEmbedding, FTCLSToken
from teras.layers import CategoricalFeatureEmbedding, Encoder
from teras.utils import get_features_metadata_for_embedding, get_tmp_dir
import keras
import pandas as pd
import numpy as np
import os
import pytest


@pytest.fixture()
def setup_data():
    data = pd.DataFrame(
        {"income": np.ones(10),
         "goals": np.ones(10),
         'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
         'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
    categorical_feats = ["player_level", "shirt_number"]
    numerical_feats = ["income", "goals"]
    features_metadata = get_features_metadata_for_embedding(
        pd.DataFrame(data),
        categorical_features=categorical_feats,
        numerical_features=numerical_feats)
    categorical_feature_embedding = CategoricalFeatureEmbedding(
        features_metadata=features_metadata,
        embedding_dim=32)
    numerical_feature_embedding = FTNumericalFeatureEmbedding(
        features_metadata=features_metadata,
        embedding_dim=32)
    cls_token = FTCLSToken(embedding_dim=32)
    encoder = Encoder()
    head = keras.layers.Dense(1)
    model = FTTransformer(input_dim=4,
                          categorical_feature_embedding=categorical_feature_embedding,
                          numerical_feature_embedding=numerical_feature_embedding,
                          cls_token=cls_token,
                          encoder=encoder,
                          head=head)
    inputs = data.values
    return inputs, model


def test_ft_transformer_valid_call(setup_data):
    inputs, model = setup_data
    model(inputs)


def test_ft_transformer_save_and_load(setup_data):
    inputs, model = setup_data
    save_path = os.path.join(get_tmp_dir(), "ft_transformer_lf.keras")
    model.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    np.testing.assert_allclose(outputs_original, outputs_reloaded)
    np.testing.assert_allclose(model.weights, reloaded_model.weights)
