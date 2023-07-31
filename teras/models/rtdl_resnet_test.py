import tensorflow as tf
from tensorflow import keras
from teras.models.rtdl_resnet import RTDLResNetClassifier, RTDLResNetRegressor
import os


class RTDLResNetClassifierTest(tf.test.TestCase):
    def test_valid_call(self):
        inputs = tf.ones((8, 5))
        model = RTDLResNetClassifier(num_classes=2,
                                     input_dim=5)
        model(inputs)

    def test_save_and_load(self):
        inputs = tf.ones((8, 5))
        model = RTDLResNetClassifier(num_classes=2,
                                     input_dim=5)
        save_path = os.path.join(self.get_temp_dir(), "rtdl_resnet_classifier.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(inputs)
        outputs_reloaded = reloaded_model(inputs)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)


class RTDLResNetRegressorTest(tf.test.TestCase):
    def test_valid_call(self):
        inputs = tf.ones((8, 5))
        model = RTDLResNetRegressor(num_outputs=2,
                                    input_dim=5)
        model(inputs)

    def test_save_and_load(self):
        inputs = tf.ones((8, 5))
        model = RTDLResNetRegressor(num_outputs=2,
                                    input_dim=5)
        save_path = os.path.join(self.get_temp_dir(), "rtdl_resnet_regressor.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(inputs)
        outputs_reloaded = reloaded_model(inputs)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)
