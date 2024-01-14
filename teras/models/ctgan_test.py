import keras
from keras import ops
from keras import random
from teras.models.ctgan import CTGANGenerator, CTGANDiscriminator, CTGAN
from teras.preprocessing.ctgan import CTGANDataTransformer, CTGANDataSampler
from teras.utils import generate_fake_gemstone_data, get_tmp_dir
import os
import pytest


@pytest.fixture
def setup_data_ctgan():
    fake_gem_df = generate_fake_gemstone_data(num_samples=32)
    num_cols = ["depth", "table"]
    cat_cols = ["cut", "color", "clarity"]
    data_transformer = CTGANDataTransformer(numerical_features=num_cols,
                                            categorical_features=cat_cols)
    x_transformed = data_transformer.transform(fake_gem_df)

    data_sampler = CTGANDataSampler(batch_size=8,
                                    categorical_features=cat_cols,
                                    metadata=data_transformer.get_metadata())
    dataset = data_sampler.get_dataset(x_transformed=x_transformed,
                                       x_original=fake_gem_df)
    metadata = data_transformer.get_metadata()
    data_dim = data_sampler.data_dim
    ctgan = CTGAN(input_dim=data_dim,
                  metadata=metadata)
    return {"dataset": dataset,
            "metadata": metadata,
            "data_dim": data_dim,
            "ctgan": ctgan,
            "data_sampler": data_sampler,
            "data_transformer": data_transformer
            }


def test_ctgan_generator_valid_call(setup_data_ctgan):
    z = random.uniform((8, 16))
    generator = CTGANGenerator(data_dim=5,
                               metadata=setup_data_ctgan["metadata"])
    outputs = generator(z)


def test_ctgan_generator_output_shape(setup_data_ctgan):
    z = random.uniform((8, 16))
    generator = CTGANGenerator(data_dim=5,
                               metadata=setup_data_ctgan["metadata"])
    outputs = generator(z)
    ops.shape(outputs) == (8, 5)


def test_ctgan_generator_save_and_load(setup_data_ctgan):
    z = random.uniform((8, 16))
    generator = CTGANGenerator(data_dim=5,
                               metadata=setup_data_ctgan["metadata"])
    save_path = os.path.join(get_tmp_dir(), "ctgan_generator.keras")
    generator.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = generator(z)
    outputs_reloaded = reloaded_model(z)
    # assertAllClose(outputs_original, outputs_reloaded)


def test_ctgan_discriminator_valid_call():
    inputs = ops.ones((8, 5))
    discriminator = CTGANDiscriminator()
    outputs = discriminator(inputs)


def test_ctgan_discriminator_output_shape():
    inputs = ops.ones((8, 5))
    discriminator = CTGANDiscriminator()
    outputs = discriminator(inputs)
    ops.shape(outputs) == (1, 1)


def test_ctgan_discriminator_save_and_load():
    inputs = ops.ones((8, 5))
    discriminator = CTGANDiscriminator()
    save_path = os.path.join(get_tmp_dir(), "ctgan_discriminator.keras")
    discriminator.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = discriminator(inputs)
    outputs_reloaded = reloaded_model(inputs)
    # assertAllClose(outputs_original, outputs_reloaded)


def test_ctgan_valid_call(setup_data_ctgan):
    # CTGAN's call method is just calls the generator's call method
    z = random.normal((8, 16))
    outputs = setup_data_ctgan["ctgan"](z)


def test_ctgan_output_shape(setup_data_ctgan):
    # CTGAN's call method is just calls the generator's call method
    z = random.normal((8, 16))
    outputs = setup_data_ctgan["ctgan"](z)
    ops.shape(outputs) == (8, setup_data_ctgan["data_dim"])


def test_ctgan_save_and_load(setup_data_ctgan):
    ctgan = setup_data_ctgan["ctgan"]
    z = random.normal((8, 16))
    save_path = os.path.join(get_tmp_dir(), "ctgan.keras")
    ctgan.save(save_path, save_format="keras_v3")
    reloaded_model = keras.models.load_model(save_path)
    outputs_original = ctgan(z)
    outputs_reloaded = reloaded_model(z)
    # assertAllClose(outputs_original, outputs_reloaded)


def test_ctgan_fit(setup_data_ctgan):
    ctgan = setup_data_ctgan["ctgan"]
    ctgan.compile()
    ctgan.fit(setup_data_ctgan["dataset"])


def test_ctgan_generate(setup_data_ctgan):
    ctgan = setup_data_ctgan["ctgan"]
    generated_data = ctgan.generate(num_samples=16,
                                    data_sampler=setup_data_ctgan["data_sampler"],
                                    data_transformer=setup_data_ctgan["data_transformer"],
                                    reverse_transform=True)
    assert len(generated_data) == 16
