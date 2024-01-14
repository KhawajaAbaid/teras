import keras
from keras import ops
from keras import random
from teras.models.pcgain import PCGAINGenerator, PCGAINDiscriminator, PCGAINClassifier, PCGAIN
from teras.preprocessing.pcgain import PCGAINDataSampler, PCGAINDataTransformer
from teras.utils import inject_missing_values, generate_fake_gemstone_data, get_tmp_dir
import os
import pytest


# ===================== PCGAIN Generator Unit Tests ====================
@pytest.fixture()
def setup_data_pcgain_generator():
    latent_inputs = random.normal((8, 16))
    return latent_inputs


def test_pcgain_generator_valid_call(setup_data_pcgain_generator):
    latent_inputs = setup_data_pcgain_generator
    generator = PCGAINGenerator(data_dim=7)
    outputs = generator(latent_inputs)


def test_pcgain_generator_output_shape(setup_data_pcgain_generator):
    latent_inputs = setup_data_pcgain_generator
    generator = PCGAINGenerator(data_dim=7)
    outputs = generator(latent_inputs)
    assert ops.shape(outputs) == (8, 7)


def test_pcgain_generator_save_and_load(setup_data_pcgain_generator):
    latent_inputs = setup_data_pcgain_generator
    generator = PCGAINGenerator(data_dim=7)
    save_path = os.path.join(get_tmp_dir(), "pcgain_generator.keras")
    generator.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = generator(latent_inputs)
    outputs_reloaded = reloaded_model(latent_inputs)
    # assertAllClose(outputs_original, outputs_reloaded)


# ===================== PCGAIN Discriminator Unit Tests ====================
@pytest.fixture()
def setup_data_pcgain_discriminator():
    inputs = ops.ones((8, 7))


def test_pcgain_discriminator_valid_call(setup_data_pcgain_discriminator):
    inputs = setup_data_pcgain_discriminator
    discriminator = PCGAINDiscriminator(data_dim=7)
    outputs = discriminator(inputs)


def test_pcgain_discriminator_output_shape(setup_data_pcgain_discriminator):
    inputs = setup_data_pcgain_discriminator
    discriminator = PCGAINDiscriminator(data_dim=7)
    outputs = discriminator(inputs)
    assert ops.shape(outputs) == (8, 7)


def test_pcgain_discriminator_save_and_load(setup_data_pcgain_discriminator):
    inputs = setup_data_pcgain_discriminator
    discriminator = PCGAINDiscriminator(data_dim=7)
    save_path = os.path.join(get_tmp_dir(), "pcgain_discriminator.keras")
    discriminator.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = discriminator(inputs)
    outputs_reloaded = reloaded_model(inputs)
    # assertAllClose(outputs_original, outputs_reloaded)


# ===================== PCGAIN Unit Tests ====================
@pytest.fixture()
def setup_data_pcgain():
    fake_gem_df = generate_fake_gemstone_data(num_samples=48)
    numerical_cols = ["depth", "table"]
    categorical_cols = ["cut", "color", "clarity"]
    x_with_missing = inject_missing_values(fake_gem_df)
    data_transformer = PCGAINDataTransformer(numerical_features=numerical_cols,
                                                  categorical_features=categorical_cols)
    x_transformed = data_transformer.fit_transform(x_with_missing, return_dataframe=True)
    x_train = x_transformed[:32]
    x_test = x_transformed[32:]
    data_sampler = PCGAINDataSampler()
    x_train = data_sampler.get_dataset(x_train)
    x_pretrain = data_sampler.get_pretraining_dataset(x_transformed, pretraining_size=0.3)
    pcgain_imputer = PCGAIN(input_dim=data_sampler.data_dim)
    return pcgain_imputer, x_pretrain, x_train, x_test, data_transformer


def test_pcgain_valid_call():
    # GAIN's call method is just calls the generator's call method
    z = random.normal((8, 16))
    pcgain = PCGAIN(input_dim=7)
    outputs = pcgain(z)


def test_pcgain_output_shape():
    inputs = random.normal((8, 16))
    pcgain = PCGAIN(input_dim=7)
    outputs = pcgain(inputs)
    assert ops.shape(outputs) == (8, 7)


def test_pcgain_save_and_load():
    inputs = random.normal((8, 16))
    pcgain = PCGAIN(input_dim=7)
    save_path = os.path.join(get_tmp_dir(), "pcgain.keras")
    pcgain.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = pcgain(inputs)
    outputs_reloaded = reloaded_model(inputs)
    # assertAllClose(outputs_original, outputs_reloaded)


def test_pcgain_fit_raises_error_if_not_pretrained_first(setup_data_pcgain):
    pcgain_imputer, _, x_train, _ = setup_data_pcgain
    with pytest.raises(AssertionError):
        pcgain_imputer.fit(x_train)


def test_pcgain_fit(setup_data_pcgain):
    pcgain_imputer, x_pretrain, x_train, _ = setup_data_pcgain
    pcgain_imputer.compile()
    pretrainer_fit_kwargs = {"epochs": 2}
    classifier_fit_kwargs = {"epochs": 2}
    pcgain_imputer.pretrain(x_pretrain, pretrainer_fit_kwargs, classifier_fit_kwargs)
    pcgain_imputer.fit(x_train)


def test_impute(setup_data_pcgain):
    pcgain_imputer, _, _, x_test, data_transformer = setup_data_pcgain
    imputed_data = pcgain_imputer.impute(x_test, data_transformer=data_transformer)
    assert imputed_data.isna().sum().sum() == 0
