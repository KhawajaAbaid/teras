import tensorflow as tf
from tensorflow import keras
from teras.models.dnfnet import DNFNetClassifier, DNFNetRegressor
import os


class DNFNetClassifierTest(tf.test.TestCase):
    def test_valid_call(self):
        inputs = tf.ones((8, 5))
        model = DNFNetClassifier(num_classes=2,
                                 num_formulas=8,
                                 input_dim=5)
        model(inputs)

    def test_save_and_load(self):
        inputs = tf.ones((8, 5))
        model = DNFNetClassifier(num_classes=2,
                                 num_formulas=8,
                                 input_dim=5)
        save_path = os.path.join(self.get_temp_dir(), "dnfnet_classifier.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(inputs)
        outputs_reloaded = reloaded_model(inputs)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)


class DNFNetRegressorTest(tf.test.TestCase):
    def test_valid_call(self):
        inputs = tf.ones((8, 5))
        model = DNFNetRegressor(num_outputs=2,
                                num_formulas=8,
                                input_dim=5)
        model(inputs)

    def test_save_and_load(self):
        inputs = tf.ones((8, 5))
        model = DNFNetRegressor(num_outputs=2,
                                num_formulas=8,
                                input_dim=5)
        save_path = os.path.join(self.get_temp_dir(), "dnfnet_regressor.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(inputs)
        outputs_reloaded = reloaded_model(inputs)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)
