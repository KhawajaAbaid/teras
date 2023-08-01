import tensorflow as tf
from tensorflow import keras
from teras.models.gain import GAINGenerator, GAINDiscriminator, GAIN
from teras.preprocessing.gain import GAINDataSampler, GAINDataTransformer
from teras.utils import inject_missing_values, generate_fake_gemstone_data
import os


class GAINGeneratorTest(tf.test.TestCase):
    def setUp(self):
        self.latent_inputs = tf.random.normal((8, 16))

    def test_valid_call(self):
        generator = GAINGenerator(data_dim=7)
        outputs = generator(self.latent_inputs)

    def test_output_shape(self):
        generator = GAINGenerator(data_dim=7)
        outputs = generator(self.latent_inputs)
        self.assertAllEqual(tf.shape(outputs), (8, 7))

    def test_save_and_load(self):
        generator = GAINGenerator(data_dim=7)
        save_path = os.path.join(self.get_temp_dir(), "gain_generator.keras")
        generator.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = generator(self.latent_inputs)
        outputs_reloaded = reloaded_model(self.latent_inputs)
        # self.assertAllClose(outputs_original, outputs_reloaded)


class GAINDiscriminatorTest(tf.test.TestCase):
    def setUp(self):
        self.inputs = tf.ones((8, 7))

    def test_valid_call(self):
        discriminator = GAINDiscriminator(data_dim=7)
        outputs = discriminator(self.inputs)

    def test_output_shape(self):
        discriminator = GAINDiscriminator(data_dim=7)
        outputs = discriminator(self.inputs)
        self.assertAllEqual(tf.shape(outputs), (8, 7))

    def test_save_and_load(self):
        discriminator = GAINDiscriminator(data_dim=7)
        save_path = os.path.join(self.get_temp_dir(), "gain_discriminator.keras")
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

        self.data_transformer = GAINDataTransformer(numerical_features=numerical_cols,
                                                    categorical_features=categorical_cols)
        x_transformed = self.data_transformer.fit_transform(x_with_missing, return_dataframe=True)
        self.x_train = x_transformed[:32]
        self.x_test = x_transformed[32:]
        self.data_sampler = GAINDataSampler()
        self.x_train = self.data_sampler.get_dataset(self.x_train)
        self.gain_imputer = GAIN(input_dim=self.data_sampler.data_dim)

    def test_valid_call(self):
        # GAIN's call method is just calls the generator's call method
        z = tf.random.normal((8, 16))
        gain = GAIN(input_dim=7)
        outputs = gain(z)

    def test_output_shape(self):
        inputs = tf.random.normal((8, 16))
        gain = GAIN(input_dim=7)
        outputs = gain(inputs)
        self.assertAllEqual(tf.shape(outputs), (8, 7))

    def test_save_and_load(self):
        inputs = tf.random.normal((8, 16))
        gain = GAIN(input_dim=7)
        save_path = os.path.join(self.get_temp_dir(), "gain.keras")
        gain.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = gain(inputs)
        outputs_reloaded = reloaded_model(inputs)
        # self.assertAllClose(outputs_original, outputs_reloaded)

    def test_fit(self):
        self.gain_imputer.compile()
        self.gain_imputer.fit(self.x_train)

    def test_impute(self):
        imputed_data = self.gain_imputer.impute(self.x_test, data_transformer=self.data_transformer)
        assert imputed_data.isna().sum().sum() == 0
