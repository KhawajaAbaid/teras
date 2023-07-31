import tensorflow as tf
from tensorflow import keras
from teras.layerflow.models.node import NODE
from teras.layers import ObliviousDecisionTree, NodeFeatureSelector
import os


class NODETest(tf.test.TestCase):
    def setUp(self):
        self.inputs = tf.ones((8, 5))
        self.input_dim = 5
        self.tree_layers = [ObliviousDecisionTree() for _ in range(3)]
        self.feature_selector = NodeFeatureSelector(data_dim=self.input_dim)
        self.head = keras.layers.Dense(1)

    def test_valid_call(self):
        model = NODE(input_dim=self.input_dim,
                     tree_layers=self.tree_layers,
                     feature_selector=self.feature_selector,
                     head=self.head)
        print(model.summary())
        model(self.inputs)

    def test_save_and_load(self):
        model = NODE(input_dim=self.input_dim,
                     tree_layers=self.tree_layers,
                     feature_selector=self.feature_selector,
                     head=self.head)
        save_path = os.path.join(self.get_temp_dir(), "node_lf.keras")
        model.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = model(self.inputs)
        outputs_reloaded = reloaded_model(self.inputs)
        self.assertAllClose(outputs_original, outputs_reloaded)
        self.assertAllClose(model.weights, reloaded_model.weights)
