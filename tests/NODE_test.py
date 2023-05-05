import keras.optimizers

from teras.models import NODEClassifier
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from sklearn import datasets as sklearn_datasets

(train_data, train_targets), (test_data, test_target) = (boston_housing.load_data())

# Regression Test
# node_regressor = NODE(n_trees=16,
#                         n_layers=8)
#
# node_regressor.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=0.05), metrics=["MAE"])
# node_regressor.fit(train_data, train_targets, batch_size=32, epochs=10)
#

# Classification Test
iris_dataset = sklearn_datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target

node_classifier = NODEClassifier(n_trees=16,
                                 n_classes=3,
                                 n_layers=8)
node_classifier.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
node_classifier.fit(X, y, batch_size=32, epochs=10)