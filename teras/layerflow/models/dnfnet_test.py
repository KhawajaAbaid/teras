import keras
from keras import ops
from teras.layerflow.models.dnfnet import DNFNet
from teras.layers.dnfnet import DNNF
from teras.utils import get_tmp_dir
import os
import numpy as np
import pytest


@pytest.fixture()
def setup_data():
    inputs = ops.ones((8, 5))
    input_dim = 5
    dnnf_layers = [DNNF(num_formulas=8) for _ in range(3)]
    head = keras.layers.Dense(1)
    model = DNFNet(input_dim=input_dim,
                   dnnf_layers=dnnf_layers,
                   head=head)
    return inputs, model


def test_dnfnet_valid_call(setup_data):
    inputs, model = setup_data
    model(inputs)


def test_dnfnet_save_and_load(setup_data):
    inputs, model = setup_data
    save_path = os.path.join(get_tmp_dir(), "dnfnet_lf.keras")
    model.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    np.testing.assert_allclose(outputs_original, outputs_reloaded)
    np.testing.assert_allclose(model.weights, reloaded_model.weights)
