import keras
import pytest
from teras.models.tabnet import TabNetClassifier, TabNetRegressor, TabNetPretrainer, TabNet
from teras.utils import get_features_metadata_for_embedding, get_tmp_dir
import pandas as pd
import numpy as np
import os
import tensorflow_probability as tfp


@pytest.fixture
def setup_data_tabnet():
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


# =================== TABNET CLASSIFIER UNITESTS ====================

def test_tabent_classifier_valid_call(setup_data_tabnet):
    inputs, features_metadata = setup_data_tabnet
    model = TabNetClassifier(num_classes=2,
                             input_dim=4,
                             features_metadata=features_metadata)
    model(inputs)


def test_tabnet_classifier_save_and_load(setup_data_tabnet):
    inputs, features_metadata = setup_data_tabnet
    model = TabNetClassifier(num_classes=2,
                             input_dim=4,
                             features_metadata=features_metadata)
    save_path = os.path.join(get_tmp_dir(), "tabnet_classifier.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)


# =================== TABNET REGRESSOR UNITESTS ====================

def test_tabnet_regressor_valid_call(setup_data_tabnet):
    inputs, features_metadata = setup_data_tabnet
    model = TabNetRegressor(num_outputs=1,
                            input_dim=4,
                            features_metadata=features_metadata)
    model(inputs)


def test_tabnet_regressor_save_and_load(setup_data_tabnet):
    inputs, features_metadata = setup_data_tabnet
    model = TabNetRegressor(num_outputs=1,
                            input_dim=4,
                            features_metadata=features_metadata)
    save_path = os.path.join(get_tmp_dir(), "tabnet_regressor.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)


# =================== TABNET PRETRAINER UNITESTS ====================

@pytest.fixture
def setup_data_tabnet_pretrainer(setup_data_tabnet):
    inputs, features_metadata = setup_data_tabnet
    binary_mask_generator = tfp.distributions.Binomial(total_count=1,
                                                       probs=0.3,
                                                       name="binary_mask_generator")
    mask = binary_mask_generator.sample(inputs.shape)
    return inputs, features_metadata, mask


def test_tabnet_pretrainer_valid_call(setup_data_tabnet_pretrainer):
    inputs, features_metadata, mask = setup_data_tabnet_pretrainer
    base_model = TabNet(input_dim=inputs.shape[1],
                        features_metadata=features_metadata)
    pretrainer = TabNetPretrainer(model=base_model,
                                  input_dim=input_dim,
                                  features_metadata=features_metadata,)
    pretrainer(inputs, mask=mask)


def test_tabnet_pretrainer_save_and_load(setup_data_tabnet_pretrainer):
    inputs, features_metadata, mask = setup_data_tabnet_pretrainer
    base_model = TabNet(input_dim=inputs.shape[1],
                        features_metadata=features_metadata)
    pretrainer = TabNetPretrainer(model=base_model,
                                  input_dim=inputs.shape[1],
                                  features_metadata=features_metadata,)
    save_path = os.path.join(get_tmp_dir(), "tabnet_pretrainer.keras")
    pretrainer.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = pretrainer(inputs, mask=mask)
    outputs_reloaded = reloaded_model(inputs, mask=mask)
