from teras._src.layers.saint.multi_head_inter_sample_attention import SAINTMultiHeadInterSampleAttention
from keras.src.testing.test_case import TestCase
from keras import ops, random


class SAINTMultiHeadInterSampleAttentionTest(TestCase):
    def setUp(self):
        self.input_shape = (8, 5, 16)
        self.input_batch = random.normal((8, 5, 16))
        self.embedding_dim = self.input_shape[-1]

    def test_valid_output_shape(self):
        inter_sample_attention = SAINTMultiHeadInterSampleAttention(
            num_heads=4,
            key_dim=self.embedding_dim//8
        )
        outputs = inter_sample_attention(self.input_batch)
        self.assertEqual(ops.shape(outputs), self.input_shape)
