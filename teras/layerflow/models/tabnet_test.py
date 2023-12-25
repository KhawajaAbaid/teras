import keras
import pytest
from keras import ops, random
from teras.layerflow.models.tabnet import (TabNet,
                                           TabNetPretrainer)
from teras.layers import (CategoricalFeatureEmbedding,
                          TabNetEncoder,
                          TabNetDecoder)
from teras.utils import get_features_metadata_for_embedding, get_tmp_dir
import pandas as pd
import numpy as np
import os


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
        embedding_dim=1)
    encoder = TabNetEncoder(data_dim=4)
    head = keras.layers.Dense(1)
    model = TabNet(
        input_dim=4,
        features_metadata=features_metadata,
        categorical_feature_embedding=categorical_feature_embedding,
        encoder=encoder,
        head=head)
    inputs = data.values
    return inputs, model


def test_tabnet_valid_call(setup_data):
    inputs, model = setup_data
    model(inputs)


def test_tabnet_save_and_load(setup_data):
    inputs, model = setup_data
    save_path = os.path.join(get_tmp_dir(), "tabnet_lf.keras")
    model.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    np.testing.assert_allclose(outputs_original, outputs_reloaded)


@pytest.fixture()
def setup_data_pretrainer():
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
        embedding_dim=1)
    encoder = TabNetEncoder(data_dim=4)
    decoder = TabNetDecoder(data_dim=4)
    inputs = data.values
    mask = random.binomial(inputs.shape,
                           counts=1,
                           probabilites=0.3)
    base_model = TabNet(
        input_dim=4,
        features_metadata=features_metadata,
        categorical_feature_embedding=categorical_feature_embedding,
        encoder=encoder)
    pretrainer = TabNetPretrainer(model=base_model,
                                  features_metadata=features_metadata,
                                  decoder=decoder,
                                  missing_feature_probability=0.3)
    return inputs, mask, base_model, pretrainer


def test_tabnet_pretrainer_valid_call(setup_data_pretrainer):
    inputs, mask, base_model, pretrainer = setup_data_pretrainer
    pretrainer(inputs, mask=mask)


def test_tabnet_pretrainer_save_and_load(setup_data_pretrainer):
    inputs, mask, base_model, pretrainer= setup_data_pretrainer
    save_path = os.path.join(get_tmp_dir(), "tabnet_pretrainer_lf.keras")
    pretrainer.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = pretrainer(inputs, mask=mask)
    outputs_reloaded = reloaded_model(inputs, mask=mask)
    # We can't check for AllClose here.. will investigate why
