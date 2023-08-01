import tensorflow as tf
from tensorflow import keras
from teras.models.ctgan import CTGANGenerator, CTGANDiscriminator, CTGAN
from teras.preprocessing.ctgan import CTGANDataTransformer, CTGANDataSampler
from teras.utils import inject_missing_values, generate_fake_gemstone_data
import os


class CTGANGeneratorTest(tf.test.TestCase):
    def setUp(self):
        fake_gem_df = generate_fake_gemstone_data(num_samples=32)
        num_cols = ["depth", "table"]
        cat_cols = ["cut", "color", "clarity"]
        self.data_transformer = CTGANDataTransformer(numerical_features=num_cols,
                                                     categorical_features=cat_cols)
        x_transformed = self.data_transformer.transform(fake_gem_df)

        self.data_sampler = CTGANDataSampler(batch_size=8,
                                             categorical_features=cat_cols,
                                             metadata=self.data_transformer.get_metadata())
        self.dataset = self.data_sampler.get_dataset(x_transformed=x_transformed,
                                                     x_original=fake_gem_df)
        self.metadata = self.data_transformer.get_metadata()

    def test_valid_call(self):
        z = tf.random.uniform((8, 16))
        generator = CTGANGenerator(data_dim=5,
                                   metadata=self.metadata)
        outputs = generator(z)

    def test_output_shape(self):
        z = tf.random.uniform((8, 16))
        generator = CTGANGenerator(data_dim=5,
                                   metadata=self.metadata)
        outputs = generator(z)
        self.assertAllEqual(tf.shape(outputs), (8, 5))

    def test_save_and_load(self):
        z = tf.random.uniform((8, 16))
        generator = CTGANGenerator(data_dim=5,
                                   metadata=self.metadata)
        save_path = os.path.join(self.get_temp_dir(), "ctgan_generator.keras")
        generator.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = generator(z)
        outputs_reloaded = reloaded_model(z)
        # self.assertAllClose(outputs_original, outputs_reloaded)


class CTGANDiscriminatorTest(tf.test.TestCase):
    def setUp(self):
        self.inputs = tf.ones((8, 5))

    def test_valid_call(self):
        discriminator = CTGANDiscriminator()
        outputs = discriminator(self.inputs)

    def test_output_shape(self):
        discriminator = CTGANDiscriminator()
        outputs = discriminator(self.inputs)
        self.assertAllEqual(tf.shape(outputs), (1, 1))

    def test_save_and_load(self):
        discriminator = CTGANDiscriminator()
        save_path = os.path.join(self.get_temp_dir(), "ctgan_discriminator.keras")
        discriminator.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = discriminator(self.inputs)
        outputs_reloaded = reloaded_model(self.inputs)
        # self.assertAllClose(outputs_original, outputs_reloaded)


class CTGANTest(tf.test.TestCase):
    def setUp(self):
        fake_gem_df = generate_fake_gemstone_data(num_samples=32)
        num_cols = ["depth", "table"]
        cat_cols = ["cut", "color", "clarity"]
        self.data_transformer = CTGANDataTransformer(numerical_features=num_cols,
                                                     categorical_features=cat_cols)
        x_transformed = self.data_transformer.transform(fake_gem_df)

        self.data_sampler = CTGANDataSampler(batch_size=8,
                                             categorical_features=cat_cols,
                                             metadata=self.data_transformer.get_metadata())
        self.dataset = self.data_sampler.get_dataset(x_transformed=x_transformed,
                                                     x_original=fake_gem_df)
        self.metadata = self.data_transformer.get_metadata()
        self.data_dim = self.data_sampler.data_dim
        self.ctgan = CTGAN(input_dim=self.data_dim,
                           metadata=self.metadata)

    def test_valid_call(self):
        # CTGAN's call method is just calls the generator's call method
        z = tf.random.normal((8, 16))
        ctgan = CTGAN(input_dim=self.data_dim,
                      metadata=self.metadata)
        outputs = ctgan(z)

    def test_output_shape(self):
        # CTGAN's call method is just calls the generator's call method
        z = tf.random.normal((8, 16))
        ctgan = CTGAN(input_dim=self.data_dim,
                      metadata=self.metadata)
        outputs = ctgan(z)
        self.assertAllEqual(tf.shape(outputs), (8, self.data_dim))

    def test_save_and_load(self):
        z = tf.random.normal((8, 16))
        ctgan = CTGAN(input_dim=self.data_dim,
                      metadata=self.metadata)
        save_path = os.path.join(self.get_temp_dir(), "ctgan.keras")
        ctgan.save(save_path, save_format="keras_v3")
        reloaded_model = keras.models.load_model(save_path)
        outputs_original = ctgan(z)
        outputs_reloaded = reloaded_model(z)
        # self.assertAllClose(outputs_original, outputs_reloaded)

    def test_fit(self):
        self.ctgan.compile()
        self.ctgan.fit(self.dataset)

    def test_generate(self):
        generated_data = self.ctgan.generate(num_samples=16,
                                             data_sampler=self.data_sampler,
                                             data_transformer=self.data_transformer,
                                             reverse_transform=True)
        assert len(generated_data) == 16
