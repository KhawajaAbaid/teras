from tensorflow import keras
import pandas as pd
from teras.models.tvae import TVAE
from teras.preprocessing.tvae import DataTransformer, DataSampler
import tensorflow as tf
tf.config.run_functions_eagerly(True)

print(f"{'-'*15} TVAE TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id", "table", "x", "y", "z"], axis=1)
cat_cols = ["cut", "color", "clarity"]
# num_cols = ["carat", "depth", "table", "x", "y", "z"]
num_cols = ["carat", "depth"]


gem_df = gem_df[:1024]

data_transformer = DataTransformer(numerical_features=num_cols,
                                   categorical_features=cat_cols)
x_transformed = data_transformer.transform(gem_df)

data_sampler = DataSampler(categorical_features=cat_cols,
                           meta_data=data_transformer.get_meta_data())
dataset = data_sampler.get_dataset(x_original=gem_df, x_transformed=x_transformed)

tvae = TVAE(data_dim=data_sampler.data_dim,
            meta_data=data_transformer.get_meta_data())

tvae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))
history = tvae.fit(dataset, epochs=2)
generated_data = tvae.generate(num_samples=1000,
                               data_transformer=data_transformer)
print(generated_data.head())
print("woah")
