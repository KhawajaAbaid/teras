import tensorflow as tf
tf.config.run_functions_eagerly(True)
import pandas as pd
from teras.models.gain import GAIN
from teras.preprocessing.gain import DataTransformer, DataSampler
from teras.utils.gain import inject_missing_values


print(f"{'-'*15}  GAIN TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)[:10000]
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

x = gem_df

x_with_missing = inject_missing_values(x)


data_transformer = DataTransformer(numerical_features=num_cols,
                                   categorical_features=cat_cols)
x_transformed = data_transformer.fit_transform(x_with_missing, return_dataframe=True)

data_sampler = DataSampler()
dataset = data_sampler.get_dataset(x_transformed)

gain_imputer = GAIN(data_sampler=data_sampler,
                    data_transformer=data_transformer)
gain_imputer.compile()
history = gain_imputer.fit(dataset, epochs=2)

test_chunk = x_transformed[500:1000]
# x_filled = gain_imputer.predict(x=test_chunk, )
# x_filled = data_transformer.reverse_transform(x_filled)
x_imputed = gain_imputer.impute(test_chunk)

print(x_imputed.head())