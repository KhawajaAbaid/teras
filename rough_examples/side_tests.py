import tensorflow as tf
import pandas as pd
from teras.utils import get_features_metadata_for_embedding

data = {'length': tf.ones(shape=(10,)),
        'area': tf.ones(shape=(10,))}
inputs_array = tf.transpose(list(data.values()))
print(inputs_array)