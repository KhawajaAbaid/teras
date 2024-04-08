from keras import ops
from teras._src.layers.cutmix import CutMix
from keras.src.testing.test_case import TestCase


class CutMixTest(TestCase):
    def setUp(self):
        self.input_batch = ops.ones((8, 4))

    def test_cutmix_valid_call(self):
        cutmix = CutMix()
        outputs = cutmix(self.input_batch)

    def test_cutmix_output_shape(self):
        cutmix = CutMix()
        outputs = cutmix(self.input_batch)
        self.assertEqual(ops.shape(outputs), ops.shape(self.input_batch))
