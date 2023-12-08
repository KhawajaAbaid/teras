import keras
from keras import ops
import numpy as np
from teras.models.rtdl_resnet import RTDLResNetClassifier, RTDLResNetRegressor
from teras.utils import get_tmp_dir
import os


def test_rtdl_resnet_classifier_valid_call():
    inputs = ops.ones((8, 5))
    model = RTDLResNetClassifier(num_classes=2,
                                 input_dim=5)
    model(inputs)


def test_rtdl_resnet_classifier_save_and_load():
    inputs = ops.ones((8, 5))
    model = RTDLResNetClassifier(num_classes=2,
                                 input_dim=5)
    save_path = os.path.join(get_tmp_dir(), "rtdl_resnet_classifier.keras")
    model.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)


def test_rtdl_resnet_regressor_valid_call():
    inputs = ops.ones((8, 5))
    model = RTDLResNetRegressor(num_outputs=2,
                                input_dim=5)
    model(inputs)


def test_rtdl_resnet_regressor_save_and_load():
    inputs = ops.ones((8, 5))
    model = RTDLResNetRegressor(num_outputs=2,
                                input_dim=5)
    save_path = os.path.join(get_tmp_dir(), "rtdl_resnet_regressor.keras")
    model.save(save_path)
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)
