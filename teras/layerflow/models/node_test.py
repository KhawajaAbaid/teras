import keras
from keras import ops
from teras.layerflow.models.node import NODE
from teras.layers import ObliviousDecisionTree, NodeFeatureSelector
from teras.utils import get_tmp_dir
import os
import numpy as np
import pytest


@pytest.fixture()
def setup_data():
    inputs = ops.ones((8, 5))
    input_dim = 5
    tree_layers = [ObliviousDecisionTree() for _ in range(3)]
    feature_selector = NodeFeatureSelector(data_dim=input_dim)
    head = keras.layers.Dense(1)
    model = NODE(input_dim=input_dim,
                 tree_layers=tree_layers,
                 feature_selector=feature_selector,
                 head=head)
    return inputs, model


def test_node_valid_call(setup_data):
    inputs, model = setup_data
    model(inputs)


def test_node_save_and_load(setup_data):
    inputs, model = setup_data
    save_path = os.path.join(get_tmp_dir(), "node_lf.keras")
    model.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    np.testing.assert_allclose(outputs_original, outputs_reloaded)
    np.testing.assert_allclose(model.weights, reloaded_model.weights)
