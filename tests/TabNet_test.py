from teras.models import TabNetClassifier, TabNetRegressor
from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
from sklearn import datasets as sklearn_datasets

(train_data, train_targets), (test_data, test_target) = (boston_housing.load_data())

# Regression Test
# tabnet_regressor = TabNetRegressor(units=64,
#                              decision_step_output_dim=64,
#                              num_decision_steps=5,
#                              relaxation_factor=1.5,
#                              batch_momentum=0.98,
#                              virtual_batch_size=None,
#                              epsilon=1e-5)
#
# tabnet_regressor.compile(loss="MSE", optimizer="Adam", metrics=["MAE"])
# tabnet_regressor.fit(train_data, train_targets, batch_size=32, epochs=10)


# Classification Test
iris_dataset = sklearn_datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target

tabnet_classifier = TabNetClassifier(units=64,
                                     num_classes=3,
                                     decision_step_output_dim=64,
                                     num_decision_steps=5,
                                     relaxation_factor=1.5,
                                     batch_momentum=0.98,
                                     virtual_batch_size=None,
                                     epsilon=1e-5)
tabnet_classifier.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="Adam",
                          metrics=[keras.metrics.SparseCategoricalAccuracy()])
tabnet_classifier.fit(X, y, batch_size=32, epochs=10)