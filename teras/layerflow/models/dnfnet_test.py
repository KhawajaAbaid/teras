import tensorflow as tf
from tensorflow import keras
from teras.layerflow.models.dnfnet import DNFNet
from teras.layers.dnfnet import DNNF
import os


class NODETest(tf.test.TestCase):
    def setUp(self):
        self.inputs = tf.ones((8, 5))
        self.input_dim = 5
        self.dnnf_layers = [DNNF(num_formulas=8) for _ in range(3)]
        self.head = keras.layers.Dense(1)

    def test_valid_call(self):
        model = DNFNet(input_dim=self.input_dim,
                       dnnf_layers=self.dnnf_layers,
                       head=self.head)
        model(self.inputs)

    def test_save_and_load(self):
        model = DNFNet(input_dim=self.input_dim,
                       dnnf_layers=self.dnnf_layers,
                       head=self.head)
        save_path = os.path.join(self.get_temp_dir(), "dnfnet_lf.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.inputs)
        outputs_reloaded = reloaded_model(self.inputs)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)
