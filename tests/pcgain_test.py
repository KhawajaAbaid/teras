import keras.optimizers.optimizer
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
import pandas as pd
from teras.models.pcgain import PCGAIN
from teras.preprocessing.pcgain import DataTransformer, DataSampler
from teras.utils.gain import inject_missing_values

from warnings import filterwarnings
filterwarnings('ignore')

print(f"{'-'*15}  PC-GAIN TEST {'-'*15}")

# Gemstone dataset from Kaggle's Playground Season 3 Episode 8
# https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
gem_df = pd.read_csv("gemstone_data/train.csv").drop(["id"], axis=1)[:10000]
cat_cols = ["cut", "color", "clarity"]
num_cols = ["carat", "depth", "table", "x", "y", "z"]

x = gem_df

x_with_missing = inject_missing_values(x)


data_transformer = DataTransformer(numerical_features=num_cols,
                                   categorical_features=cat_cols)
x_transformed = data_transformer.transform(x_with_missing, return_dataframe=True)

data_sampler = DataSampler()
dataset = data_sampler.get_dataset(x_transformed)
pretraining_dataset = data_sampler.get_pretraining_dataset(x_transformed, pretraining_size=0.4)

pcgain_imputer = PCGAIN()
pcgain_imputer.compile()
# You MUST pretrain first or perish
pretrainer_fit_kwargs = {"epochs": 2}
classifier_fit_kwargs = {"epochs": 2}
pcgain_imputer.pretrain(pretraining_dataset, pretrainer_fit_kwargs, classifier_fit_kwargs)
history = pcgain_imputer.fit(dataset, epochs=2)


test_chunk = x_transformed[500:1000]
x_filled = pcgain_imputer.predict(x=test_chunk)
x_filled = data_transformer.reverse_transform(x_filled)
print(x_filled.head())