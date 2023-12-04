import keras
import pandas as pd
import numpy as np
from teras.models.saint import SAINTClassifier, SAINTRegressor, SAINT, SAINTPretrainer
from teras.utils import get_features_metadata_for_embedding, get_tmp_dir
import os
import pytest


@pytest.fixture()
def setup_data_saint():
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


def test_sain_classifier_valid_call(setup_data_saint):
    inputs, features_metadata = setup_data_saint
    model = SAINTClassifier(features_metadata=features_metadata,
                            input_dim=4)
    model(inputs)


def test_saint_classifier_save_and_load(setup_data_saint):
    inputs, features_metadata = setup_data_saint
    model = SAINTClassifier(features_metadata=features_metadata,
                            input_dim=4)
    save_path = os.path.join(get_tmp_dir(), "saint_classifier.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)


def test_saint_regressor_valid_call(setup_data_saint):
    inputs, features_metadata = setup_data_saint
    model = SAINTRegressor(features_metadata=features_metadata,
                           input_dim=4)
    model(inputs)


def test_saint_regressor_save_and_load(setup_data_saint):
    inputs, features_metadata = setup_data_saint
    model = SAINTRegressor(features_metadata=features_metadata,
                           input_dim=4)
    save_path = os.path.join(get_tmp_dir(), "saint_regressor.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)


def test_saint_pretrainer_valid_call(setup_data_saint):
    inputs, features_metadata = setup_data_saint
    base_model = SAINT(features_metadata=features_metadata,
                       input_dim=4)
    pretrainer = SAINTPretrainer(model=base_model,
                                 features_metadata=features_metadata)
    pretrainer(inputs)


def test_saint_pretrainer_save_and_load(setup_data_saint):
    inputs, features_metadata = setup_data_saint
    base_model = SAINT(features_metadata=features_metadata,
                       input_dim=4)
    pretrainer = SAINTPretrainer(model=base_model,
                                 features_metadata=features_metadata)
    save_path = os.path.join(get_tmp_dir(), "saint_pretrainer.keras")
    pretrainer.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = pretrainer(inputs)
    outputs_reloaded = reloaded_model(inputs)
