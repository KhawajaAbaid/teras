from keras import ops
from teras._src.layers.mixup import MixUp
from keras.src.testing.test_case import TestCase


class MixUpTest(TestCase):
    def setUp(self):
        self.input_batch = ops.ones((8, 4))

    def test_mixup_valid_call(self):
        mixup = MixUp()
        outputs = mixup(self.input_batch)

    def test_mixup_output_shape(self):
        mixup = MixUp()
        outputs = mixup(self.input_batch)
        self.assertEqual(ops.shape(outputs),
                         ops.shape(self.input_batch))
