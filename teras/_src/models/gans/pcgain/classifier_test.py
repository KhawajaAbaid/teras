import os

import keras
from keras import ops
from keras.src.testing.test_case import TestCase

from teras._src.models.gans.pcgain.classifier import PCGAINClassifier


class PCGAINClassifierTest(TestCase):
    def setUp(self):
        self.inputs = ops.ones((8, 16))

    def test_valid_call(self):
        classifier = PCGAINClassifier(num_classes=5,
                                      data_dim=7)
        outputs = classifier(self.inputs)

    def test_valid_output_shape(self):
        classifier = PCGAINClassifier(num_classes=5,
                                      data_dim=7)
        outputs = classifier(self.inputs)
        assert ops.shape(outputs) == (8, 5)

    def test_save_and_load(self):
        classifier = PCGAINClassifier(num_classes=5,
                                      data_dim=7)
        outputs_original = classifier(self.inputs)
        save_path = os.path.join(self.get_temp_dir(),
                                 "pcgain_classifier.keras")
        classifier.save(save_path)
        reloaded_model = keras.models.load_model(save_path)
        # Check we got the real object back
        self.assertIsInstance(reloaded_model, PCGAINClassifier)
        outputs_reloaded = reloaded_model(self.inputs)
        self.assertAllClose(
            ops.convert_to_numpy(outputs_original),
            ops.convert_to_numpy(outputs_reloaded)
        )
