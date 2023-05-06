from tensorflow import keras
from teras.models import RTDLResNetClassifier
from teras.models import RTDLResNetRegressor
from tensorflow.keras.datasets import boston_housing
from sklearn import datasets as sklearn_datasets



# ------------------ REGRESSION TEST ----------------------

# (train_data, train_targets), (test_data, test_target) = (boston_housing.load_data())
# mean  = train_data.mean(axis=0)
# std = train_data.std(axis=0)
#
# train_data -= mean
# train_data /= std
#
# resnet_regressor = RTDLResNetRegressor(num_blocks=8,
#                                        main_dim=16,
#                                        hidden_dim=32)
#
# resnet_regressor.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=0.05), metrics=["MAE"])
# resnet_regressor.fit(train_data, train_targets, batch_size=32, epochs=10)


# ------------------ CLASSIFICATION TEST ----------------
# Classification Test
iris_dataset = sklearn_datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target

resnet_classifier = RTDLResNetClassifier(num_classes=3,
                                       num_blocks=8,
                                       hidden_dim=16,
                                       main_dim=8)
resnet_classifier.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                        optimizer=keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["accuracy"])
resnet_classifier.fit(X, y, batch_size=32, epochs=10)