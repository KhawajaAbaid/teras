import keras
from keras import ops, random
from teras.layerflow.models.tabtransformer import (
    TabTransformer,
    TabTransformerPretrainer)
from teras.layers import TabTransformerColumnEmbedding
from teras.layers import CategoricalFeatureEmbedding, Encoder
from teras.utils import get_features_metadata_for_embedding, get_tmp_dir
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
    numerical_feature_normalization = keras.layers.LayerNormalization()
    column_embedding = TabTransformerColumnEmbedding(
        num_categorical_features=2)
    encoder = Encoder()
    head = keras.layers.Dense(1)
    model = TabTransformer(
        input_dim=4,
        categorical_feature_embedding=categorical_feature_embedding,
        column_embedding=column_embedding,
        numerical_feature_normalization=numerical_feature_normalization,
        encoder=encoder,
        head=head)
    inputs = data.values
    return inputs, model


def test_tabtransformer_valid_call(setup_data):
    inputs, model = setup_data
    model(inputs)


def test_tabtransformer_save_and_load(setup_data):
    inputs, model = setup_data
    save_path = os.path.join(get_tmp_dir(), "tabtransformer_lf.keras")
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
        embedding_dim=32)
    numerical_feature_normalization = keras.layers.LayerNormalization()
    column_embedding = TabTransformerColumnEmbedding(
        num_categorical_features=2)
    encoder = Encoder()
    inputs = data.values
    num_features = 4
    num_features_to_replace = 2
    feature_indices_to_replace = random.uniform(
        shape=(ops.shape(inputs)[0], num_features_to_replace),
        maxval=num_features,
        dtype="int32")
    mask = ops.max(ops.one_hot(feature_indices_to_replace,
                               num_classes=num_features),
                   axis=1)
    base_model = TabTransformer(
        input_dim=4,
        categorical_feature_embedding=categorical_feature_embedding,
        column_embedding=column_embedding,
        numerical_feature_normalization=numerical_feature_normalization,
        encoder=encoder)
    pretrainer = TabTransformerPretrainer(
        model=base_model,
        features_metadata=features_metadata)
    return inputs, mask, base_model, pretrainer


def test_tabtransformer_pretrainer_valid_call(setup_data_pretrainer):
    inputs, mask, base_model, pretrainer = setup_data_pretrainer
    pretrainer(inputs, mask=mask)


def test_tabtransformer_pretrainer_save_and_load(setup_data_pretrainer):
    inputs, mask, base_model, pretrainer = setup_data_pretrainer
    save_path = os.path.join(get_tmp_dir(),
                             "tabtransformer_pretrainer_lf.keras")
    pretrainer.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = pretrainer(inputs, mask=mask)
    outputs_reloaded = reloaded_model(inputs, mask=mask)
    # We can't check for AllClose because the call method randomly shuffles the inputs
