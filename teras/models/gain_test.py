import keras
from keras import ops
from keras import random
from teras.models.gain import GAINGenerator, GAINDiscriminator, GAIN
from teras.preprocessing.gain import GAINDataSampler, GAINDataTransformer
from teras.utils import inject_missing_values, generate_fake_gemstone_data, get_tmp_dir
import os
import pytest


# =================== GAIN GENERATOR UNIT TESTS ==================

@pytest.fixture()
def setup_data():
    latent_inputs = random.normal((8, 16))
    return latent_inputs


def test_gan_generator_valid_call(setup_data):
    latent_inputs = setup_data
    generator = GAINGenerator(data_dim=7)
    outputs = generator(latent_inputs)


def test_gan_generator_output_shape(setup_data):
    latent_inputs = setup_data
    generator = GAINGenerator(data_dim=7)
    outputs = generator(latent_inputs)
    assert ops.shape(outputs) == (8, 7)


def test_gan_generator_save_and_load(setup_data):
    latent_inputs = setup_data
    generator = GAINGenerator(data_dim=7)
    save_path = os.path.join(get_tmp_dir(), "gain_generator.keras")
    generator.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = generator(latent_inputs)
    outputs_reloaded = reloaded_model(latent_inputs)
    # assertAllClose(outputs_original, outputs_reloaded)


# =================== GAIN DISCRIMINATOR UNIT TESTS ==================

@pytest.fixture()
def setup_data_disc():
    return ops.ones((8, 7))


def test_gan_discriminator_valid_call(setup_data_disc):
    inputs = setup_data_disc
    discriminator = GAINDiscriminator(data_dim=7)
    outputs = discriminator(inputs)


def test_gan_discriminator_output_shape(setup_data_disc):
    inputs = setup_data_disc
    discriminator = GAINDiscriminator(data_dim=7)
    outputs = discriminator(inputs)
    assert ops.shape(outputs) == (8, 7)


def test_gan_discriminator_save_and_load(setup_data_disc):
    inputs = setup_data_disc
    discriminator = GAINDiscriminator(data_dim=7)
    save_path = os.path.join(get_tmp_dir(), "gain_discriminator.keras")
    discriminator.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = discriminator(inputs)
    outputs_reloaded = reloaded_model(inputs)
    # assertAllClose(outputs_original, outputs_reloaded)


# =================== GAIN  UNIT TESTS ==================

@pytest.fixture()
def setup_data_gain():
    fake_gem_df = generate_fake_gemstone_data(num_samples=48)
    numerical_cols = ["depth", "table"]
    categorical_cols = ["cut", "color", "clarity"]
    x_with_missing = inject_missing_values(fake_gem_df)

    data_transformer = GAINDataTransformer(numerical_features=numerical_cols,
                                                categorical_features=categorical_cols)
    x_transformed = data_transformer.fit_transform(x_with_missing, return_dataframe=True)
    x_train = x_transformed[:32]
    x_test = x_transformed[32:]
    data_sampler = GAINDataSampler()
    x_train = data_sampler.get_dataset(x_train)
    gain_imputer = GAIN(input_dim=data_sampler.data_dim)
    return gain_imputer, x_train, x_test, data_transformer


def test_gain_valid_call():
    # GAIN's call method is just calls the generator's call method
    z = random.normal((8, 16))
    gain = GAIN(input_dim=7)
    outputs = gain(z)


def test_gain_output_shape():
    inputs = random.normal((8, 16))
    gain = GAIN(input_dim=7)
    outputs = gain(inputs)
    assert ops.shape(outputs) == (8, 7)


def test_gain_save_and_load():
    inputs = ops.random.normal((8, 16))
    gain = GAIN(input_dim=7)
    save_path = os.path.join(get_tmp_dir(), "gain.keras")
    gain.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = gain(inputs)
    outputs_reloaded = reloaded_model(inputs)
    # assertAllClose(outputs_original, outputs_reloaded)


def test_gain_fit(setup_data_gain):
    gain_imputer, x_train, _ = setup_data_gain
    gain_imputer.compile()
    gain_imputer.fit(x_train)


def test_gain_impute():
    gain_imputer, _, x_test, data_transformer = setup_data_gain
    imputed_data = gain_imputer.impute(x_test, data_transformer=data_transformer)
    assert imputed_data.isna().sum().sum() == 0
