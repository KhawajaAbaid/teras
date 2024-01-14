import keras
from teras.layerflow.models.saint import (SAINT,
                                          SAINTPretrainer)
from teras.layers import (CategoricalFeatureEmbedding,
                          SAINTNumericalFeatureEmbedding,
                          SAINTEncoder,
                          MixUp,
                          CutMix,
                          SAINTReconstructionHead,
                          SAINTProjectionHead)
from teras.utils import get_features_metadata_for_embedding, get_tmp_dir
import pandas as pd
import numpy as np
import os
import pytest


@pytest.fixture()
def setup_data():
    data = pd.DataFrame({"income": np.ones(10),
                         "goals": np.ones(10),
                         'player_level': [5, 7, 9, 8, 9, 10, 9, 7, 8, 9],
                         'shirt_number': [7, 10, 10, 7, 7, 10, 10, 10, 7, 10]})
    categorical_feats = ["player_level", "shirt_number"]
    numerical_feats = ["income", "goals"]
    input_dim = 4
    features_metadata = get_features_metadata_for_embedding(
        pd.DataFrame(data),
        categorical_features=categorical_feats,
        numerical_features=numerical_feats)
    categorical_feature_embedding = CategoricalFeatureEmbedding(
        features_metadata=features_metadata,
        embedding_dim=32)
    numerical_feature_embedding = SAINTNumericalFeatureEmbedding(
        features_metadata=features_metadata,
        embedding_dim=32)
    encoder = SAINTEncoder(data_dim=input_dim)
    head = keras.layers.Dense(1)
    model = SAINT(
        input_dim=input_dim,
        categorical_feature_embedding=categorical_feature_embedding,
        numerical_feature_embedding=numerical_feature_embedding,
        encoder=encoder,
        head=head)
    inputs = data.values
    return inputs, model


def test_saint_valid_call(setup_data):
    inputs, model = setup_data
    model(inputs)


def test_saint_save_and_load(setup_data):
    inputs, model = setup_data
    save_path = os.path.join(get_tmp_dir(), "saint_lf.keras")
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
    input_dim = 4
    features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                            categorical_features=categorical_feats,
                                                            numerical_features=numerical_feats)
    categorical_feature_embedding = CategoricalFeatureEmbedding(features_metadata=features_metadata,
                                                                embedding_dim=32)
    numerical_feature_embedding = SAINTNumericalFeatureEmbedding(features_metadata=features_metadata,
                                                                 embedding_dim=32)
    encoder = SAINTEncoder(data_dim=input_dim)
    # layers for pretrainer
    mixup = MixUp()
    cutmix = CutMix()
    projection_head_1 = SAINTProjectionHead()
    projection_head_2 = SAINTProjectionHead()
    reconstruction_head = SAINTReconstructionHead(features_metadata=features_metadata)
    base_model = SAINT(
        input_dim=input_dim,
        categorical_feature_embedding=categorical_feature_embedding,
        numerical_feature_embedding=numerical_feature_embedding,
        encoder=encoder)
    pretrainer = SAINTPretrainer(
        model=base_model,
        features_metadata=features_metadata,
        mixup=mixup,
        cutmix=cutmix,
        projection_head_1=projection_head_1,
        projection_head_2=projection_head_2,
        reconstruction_head=reconstruction_head)
    inputs = data.values
    return inputs, base_model, pretrainer


def test_saint_pretrainer_valid_call(setup_data_pretrainer):
    inputs, base_model, pretrainer = setup_data_pretrainer
    pretrainer(inputs)


def test_saint_pretrainer_save_and_load(setup_data_pretrainer):
    inputs, base_model, pretrainer = setup_data_pretrainer
    save_path = os.path.join(get_tmp_dir(), "saint_pretrainer_lf.keras")
    pretrainer.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = pretrainer(inputs)
    outputs_reloaded = reloaded_model(inputs)
    # We can't check for AllClose
