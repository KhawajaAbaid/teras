from tensorflow import keras
import pandas as pd
from teras.generative import CTGAN
from teras.preprocessing.ctgan import CTGANDataSampler, CTGANDataTransformer
import tensorflow as tf
# tf.config.run_functions_eagerly(True)

print(f"{'-'*15}  CTGAN TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id", "table", "x", "y", "z"], axis=1)
cat_cols = ["cut", "color", "clarity"]
# num_cols = ["carat", "depth", "table", "x", "y", "z"]
num_cols = ["carat", "depth"]


gem_df = gem_df[:1024]

data_transformer = CTGANDataTransformer(numerical_features=num_cols,
                                        categorical_features=cat_cols)
x_transformed = data_transformer.transform(gem_df)

data_sampler = CTGANDataSampler(batch_size=512,
                                categorical_features=cat_cols,
                                metadata=data_transformer.get_metadata())
dataset = data_sampler.get_dataset(x_transformed=x_transformed,
                                   x_original=gem_df)

ctgan = CTGAN(input_dim=data_sampler.data_dim,
              metadata=data_transformer.get_metadata())
ctgan.compile()
history = ctgan.fit(dataset, epochs=2)
generated_data = ctgan.generate(num_samples=1000,
                                data_sampler=data_sampler,
                                data_transformer=data_transformer,
                                reverse_transform=True)
print(generated_data.head())