import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List
import numpy as np

INT_OR_FLOAT = List[int, float]

# NOTE: This is a very simple implementation of Bagging.
# And mostly is for demonstration purpose.
# Will make it efficient and better soon enough.
# Also, currently there's just Bagging class, but my plan is to add
# BaggingClassifier and BaggingRegressor like sklearn.


class Bagging:
    def __int__(self,
                build_keras_model_func = None,
                num_estimators: int = 5,
                max_samples: INT_OR_FLOAT = 1.0,
                max_features: INT_OR_FLOAT = 1.0,
                bootstrap: bool = True,
                oob_score: bool = True):
        """
        Args:
             build_keras_model_func: A function that builds and returns Keras model.
        """
        self.build_keras_model_func = build_keras_model_func
        self.num_estimators = num_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.trained_models = []
        self.all_predictions = []

    def fit(self, X, y):
        for i in range(self.num_estimators):
            bootstrap_indices = np.random.randint(0, len(X), size=len(y))
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            model = self.build_keras_model_func()
            model.fit(X_bootstrap, y_bootstrap)
            self.trained_models.append(model)

    def predict(self, X_test):
        for model in self.trained_models:
            preds = model.predict(X_test)
            self.all_predictions.append(preds)
        return np.mean(self.all_predictions, axis=-1)