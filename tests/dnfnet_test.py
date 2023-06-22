from tensorflow import keras
from teras.models import DNFNetRegressor, DNFNetClassifier
from tensorflow.keras.datasets import boston_housing
import tensorflow as tf


# PLEASE NOTE: The training does NOT get stuck, it just takes a while to start
# because of the complex tensorflow graph building. But once it starts, it literally zooms.

# <<<<<<<<<<<<<<<<<<<<< REGRESSION TEST >>>>>>>>>>>>>>>>>>>>>>>>>>>
print(F"{'-'*15} REGRESSSION TEST {'-'*15}")
(train_data, train_targets), (test_data, test_target) = (boston_housing.load_data())
# train_targets = tf.cast(train_targets, dtype="float32")
# test_target = tf.cast(test_target, dtype="float32")

dnf_model = DNFNetRegressor(n_formulas=256)

dnf_model.compile(loss=keras.losses.MSE, metrics=keras.metrics.MAE,
                  optimizer=keras.optimizers.Adam(learning_rate=0.05))

dnf_model.fit(train_data, train_targets, batch_size=32, epochs=10)


# <<<<<<<<<<<<<<<<<<<<< CLASSIFICATION TEST >>>>>>>>>>>>>>>>>>>>>>>>>>>
print(F"{'-'*15} CLASSIFICAITON TEST {'-'*15}")
from sklearn import datasets as sklearn_datasets

iris_dataset = sklearn_datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target

dnf_classifier = DNFNetClassifier(n_formulas=256,
                                  num_dnnf_layers=3,
                                     num_classes=3)
dnf_classifier.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer="Adam",
                          metrics=keras.metrics.SparseCategoricalAccuracy())
dnf_classifier.fit(X, y, batch_size=32, epochs=10)