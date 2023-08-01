import tensorflow as tf
from tensorflow import keras
from teras.models.pcgain import PCGAINGenerator, PCGAINDiscriminator, PCGAINClassifier, PCGAIN
from teras.preprocessing.pcgain import PCGAINDataSampler, PCGAINDataTransformer
from teras.utils import inject_missing_values, generate_fake_gemstone_data
import os
import pytest


class PCGAINGeneratorTest(tf.test.TestCase):
    def setUp(self):
        self.latent_inputs = tf.random.normal((8, 16))

    def test_valid_call(self):
        generator = PCGAINGenerator(data_dim=7)
        outputs = generator(self.latent_inputs)

    def test_output_shape(self):
        generator = PCGAINGenerator(data_dim=7)
        outputs = generator(self.latent_inputs)
        self.assertAllEqual(tf.shape(outputs), (8, 7))

    def test_save_and_load(self):
        generator = PCGAINGenerator(data_dim=7)
        save_path = os.path.join(self.get_temp_dir(), "pcgain_generator.keras")
        generator.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = generator(self.latent_inputs)
        outputs_reloaded = reloaded_model(self.latent_inputs)
        # self.assertAllClose(outputs_original, outputs_reloaded)


class GAINDiscriminatorTest(tf.test.TestCase):
    def setUp(self):
        self.inputs = tf.ones((8, 7))

    def test_valid_call(self):
        discriminator = PCGAINDiscriminator(data_dim=7)
        outputs = discriminator(self.inputs)

    def test_output_shape(self):
        discriminator = PCGAINDiscriminator(data_dim=7)
        outputs = discriminator(self.inputs)
        self.assertAllEqual(tf.shape(outputs), (8, 7))

    def test_save_and_load(self):
        discriminator = PCGAINDiscriminator(data_dim=7)
        save_path = os.path.join(self.get_temp_dir(), "pcgain_discriminator.keras")
        discriminator.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = discriminator(self.inputs)
        outputs_reloaded = reloaded_model(self.inputs)
        # self.assertAllClose(outputs_original, outputs_reloaded)


class GAINTest(tf.test.TestCase):
    def setUp(self):
        fake_gem_df = generate_fake_gemstone_data(num_samples=48)
        numerical_cols = ["depth", "table"]
        categorical_cols = ["cut", "color", "clarity"]
        x_with_missing = inject_missing_values(fake_gem_df)
        self.data_transformer = PCGAINDataTransformer(numerical_features=numerical_cols,
                                                      categorical_features=categorical_cols)
        x_transformed = self.data_transformer.fit_transform(x_with_missing, return_dataframe=True)
        self.x_train = x_transformed[:32]
        self.x_test = x_transformed[32:]
        self.data_sampler = PCGAINDataSampler()
        self.x_train = self.data_sampler.get_dataset(self.x_train)
        self.x_pretrain = self.data_sampler.get_pretraining_dataset(x_transformed, pretraining_size=0.3)
        self.pcgain_imputer = PCGAIN(input_dim=self.data_sampler.data_dim)
        self.pcgain_imputer.compile()

    def test_valid_call(self):
        # GAIN's call method is just calls the generator's call method
        z = tf.random.normal((8, 16))
        pcgain = PCGAIN(input_dim=7)
        outputs = pcgain(z)

    def test_output_shape(self):
        inputs = tf.random.normal((8, 16))
        pcgain = PCGAIN(input_dim=7)
        outputs = pcgain(inputs)
        self.assertAllEqual(tf.shape(outputs), (8, 7))

    def test_save_and_load(self):
        inputs = tf.random.normal((8, 16))
        pcgain = PCGAIN(input_dim=7)
        save_path = os.path.join(self.get_temp_dir(), "pcgain.keras")
        pcgain.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = pcgain(inputs)
        outputs_reloaded = reloaded_model(inputs)
        # self.assertAllClose(outputs_original, outputs_reloaded)

    def test_fit_raises_error_if_not_pretrained_first(self):
        with pytest.raises(AssertionError):
            self.pcgain_imputer.fit(self.x_train)

    def test_fit(self):
        self.pcgain_imputer.compile()
        pretrainer_fit_kwargs = {"epochs": 2}
        classifier_fit_kwargs = {"epochs": 2}
        self.pcgain_imputer.pretrain(self.x_pretrain, pretrainer_fit_kwargs, classifier_fit_kwargs)
        self.pcgain_imputer.fit(self.x_train)

    def test_impute(self):
        imputed_data = self.pcgain_imputer.impute(self.x_test, data_transformer=self.data_transformer)
        assert imputed_data.isna().sum().sum() == 0
