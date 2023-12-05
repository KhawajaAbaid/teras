import keras
from keras import ops
from teras.models import (FTTransformerClassifier,
                          FTTransformerRegressor)
from teras.utils import get_features_metadata_for_embedding, get_tmp_dir
import pandas as pd
import numpy as np
import os
import pytest


@pytest.fixture
def setup_data_fttransformer():
    data = pd.DataFrame({"income": np.ones(10),
                         "goals": np.ones(10),
                         'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                         'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
    categorical_feats = ["player_level", "shirt_number"]
    numerical_feats = ["income", "goals"]
    features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                                 categorical_features=categorical_feats,
                                                                 numerical_features=numerical_feats)
    return data.values, features_metadata


def test_ft_transfromer_classifier_valid_call(setup_data_fttransformer):
    inputs, features_metadata = setup_data_fttransformer
    model = FTTransformerClassifier(num_classes=2,
                                    input_dim=4,
                                    features_metadata=features_metadata)
    model(inputs)


def test_ft_transfromer_classifier_save_and_load(setup_data_fttransformer):
    inputs, features_metadata = setup_data_fttransformer
    model = FTTransformerClassifier(num_classes=2,
                                    input_dim=4,
                                    features_metadata=features_metadata)
    save_path = os.path.join(get_tmp_dir(), "ft_transformer_classifier.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)


def test_ft_transformer_regressor_valid_call(setup_data_fttransformer):
    inputs, features_metadata = setup_data_fttransformer
    model = FTTransformerRegressor(num_outputs=1,
                                   input_dim=4,
                                   features_metadata=features_metadata)
    model(inputs)


def test_ft_transformer_regressor_save_and_load(setup_data_fttransformer):
    inputs, features_metadata = setup_data_fttransformer
    model = FTTransformerRegressor(num_outputs=1,
                                   input_dim=4,
                                   features_metadata=features_metadata)
    save_path = os.path.join(get_tmp_dir(), "ft_transformer_regressor.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
