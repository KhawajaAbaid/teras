import keras
from keras import ops
from teras.layerflow.models.rtdl_resnet import RTDLResNet
from teras.layers import RTDLResNetBlock
from teras.utils import get_tmp_dir
import os
import pytest
import numpy as np


@pytest.fixture
def setup_data_rtdl_resnet():
    inputs = ops.ones((8, 5))
    resnet_blocks = [RTDLResNetBlock() for _ in range(3)]
    head = keras.layers.Dense(1)
    model = RTDLResNet(input_dim=inputs.shape[1],
                       resnet_blocks=resnet_blocks,
                       head=head)
    return inputs, model


def test_valid_call(setup_data_rtdl_resnet):
    inputs, model = setup_data_rtdl_resnet
    model(inputs)


def test_save_and_load(setup_data_rtdl_resnet):
    inputs, model = setup_data_rtdl_resnet
    save_path = os.path.join(get_tmp_dir(), "rtdl_resnet_lf.keras")
    model.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)
