import tensorflow as tf
from tensorflow import keras
from teras.layerflow.models.rtdl_resnet import RTDLResNet
from teras.layers import RTDLResNetBlock
import os


class NODETest(tf.test.TestCase):
    def setUp(self):
        self.inputs = tf.ones((8, 5))
        self.input_dim = 5
        self.resnet_blocks = [RTDLResNetBlock() for _ in range(3)]
        self.head = keras.layers.Dense(1)

    def test_valid_call(self):
        model = RTDLResNet(input_dim=self.input_dim,
                           resnet_blocks=self.resnet_blocks,
                           head=self.head)
        model(self.inputs)

    def test_save_and_load(self):
        model = RTDLResNet(input_dim=self.input_dim,
                           resnet_blocks=self.resnet_blocks,
                           head=self.head)
        save_path = os.path.join(self.get_temp_dir(), "rtdl_resnet_lf.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.inputs)
        outputs_reloaded = reloaded_model(self.inputs)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)
