import keras
from keras import ops
from teras.models.dnfnet import DNFNetClassifier, DNFNetRegressor
from teras.utils import get_tmp_dir
import os
import numpy as np


def test_dnfnet_classifier_valid_call():
    inputs = ops.ones((8, 5))
    model = DNFNetClassifier(num_classes=2,
                             num_formulas=8,
                             input_dim=5)
    model(inputs)


def test_dnfnet_classifier_save_and_load():
    inputs = ops.ones((8, 5))
    model = DNFNetClassifier(num_classes=2,
                             num_formulas=8,
                             input_dim=5)
    save_path = os.path.join(get_tmp_dir(), "dnfnet_classifier.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)


def test_dnfnet_regressor_valid_call():
    inputs = ops.ones((8, 5))
    model = DNFNetRegressor(num_outputs=2,
                            num_formulas=8,
                            input_dim=5)
    model(inputs)


def test_dnfnet_regressor_save_and_load():
    inputs = ops.ones((8, 5))
    model = DNFNetRegressor(num_outputs=2,
                            num_formulas=8,
                            input_dim=5)
    save_path = os.path.join(get_tmp_dir(), "dnfnet_regressor.keras")
    model.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = model(inputs)
    outputs_reloaded = reloaded_model(inputs)
    assert np.allclose(outputs_original, outputs_reloaded)
    assert np.allclose(model.weights, reloaded_model.weights)
