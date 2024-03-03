import keras
from keras import random, ops
from teras.models.pretrainers.tabnet import TabNetPretrainer
from teras.models.backbones.tabnet.encoder import TabNetEncoderBackbone
from teras.models.tabnet.decoder import TabNetDecoder
from keras.src.testing.test_case import TestCase


class TabNetPretrainerTest(TestCase):
    def setUp(self):
        self.batch_size = 16
        self.input_batch = random.normal((self.batch_size, 5))
        self.mask = random.binomial((self.batch_size, 5),
                                    1, 0.5)

    def test_valid_call(self):
        encoder = TabNetEncoderBackbone(
            input_dim=5,
            feature_transformer_dim=16,
            decision_step_dim=8)
        decoder = TabNetDecoder(data_dim=5,
                                feature_transformer_dim=16,
                                decision_step_dim=8)
        pretrainer = TabNetPretrainer(
            encoder=encoder,
            decoder=decoder
        )
        reconstructed_features = pretrainer(self.input_batch,
                                            self.mask)

    def test_valid_output_shape(self):
        encoder = TabNetEncoderBackbone(
            input_dim=5,
            feature_transformer_dim=16,
            decision_step_dim=16)
        decoder = TabNetDecoder(data_dim=5,
                                feature_transformer_dim=16,
                                decision_step_dim=16)
        pretrainer = TabNetPretrainer(
            encoder=encoder,
            decoder=decoder
        )
        reconstructed_features = pretrainer(self.input_batch,
                                            self.mask)
        self.assertEqual(ops.shape(reconstructed_features),
                         (self.batch_size, 5))

    def test_fit(self):
        encoder = TabNetEncoderBackbone(
            input_dim=5,
            feature_transformer_dim=16,
            decision_step_dim=16)
        decoder = TabNetDecoder(data_dim=5,
                                feature_transformer_dim=16,
                                decision_step_dim=16)
        pretrainer = TabNetPretrainer(
            encoder=encoder,
            decoder=decoder
        )
        pretrainer.compile(optimizer=keras.optimizers.Adam())
        pretrainer.build(ops.shape(self.input_batch))
        pretrainer.fit(self.input_batch)
