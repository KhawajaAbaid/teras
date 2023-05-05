import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from typing import List

MODELS_LIST = List[keras.Model]

# NOTE: This is a very simple implementation of Bagging.
# And mostly is for demonstration purpose.
# Will make it efficient and better soon enough.
# Also, currently there's just Bagging class, but my plan is to add
# BaggingClassifier and BaggingRegressor like sklearn.


class Stacking:
    def __int__(self,
                base_learners: MODELS_LIST,
                num_splits: int = 5
                ):
        self.base_learners = base_learners
        self.num_splits = num_splits

    def fit(self, X, y):
        self.meta_data = np.zeros((len(self.base_learners), len(X)))
        self.meta_targets = np.zeros(len(X))

        kf = KFold(n_splits=self.num_splits, shuffle=True)
        meta_index = 0

        for train_idx, test_idx in kf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            for i in range(len(self.base_learners)):
                learner = self.base_learners[i]
                learner.fit(X_train, y_train)
                predictions = learner.predict(X_test)
                self.meta_data[i][meta_index:meta_index + len(test_idx)] = predictions

            self.meta_targets[meta_index:meta_index + len(test_idx)] = y_test
            meta_index += len(test_idx)

        return self.meta_data, self.meta_targets