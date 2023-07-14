from teras.layerflow.models.tabnet import TabNetClassifier, TabNetRegressor
from teras.utils import get_features_metadata_for_embedding
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def setup_data():
    data = {"length": np.ones(10),
            "area": np.ones(10),
            "color": ['green', 'yellow', 'green', 'orange', 'red',
                      'orange', 'orange', 'green', 'red', 'red'],
            "shape": ["circle", "square", "square", "rectangle", "circle",
                      "square", "circle", "rectangle", "circle", "square"]}
    categorical_feats = ["color", "shape"]
    numerical_feats = ["length", "area"]
    features_metadata = get_features_metadata_for_embedding(pd.DataFrame(data),
                                                            categorical_features=categorical_feats,
                                                            numerical_features=numerical_feats)
    return data, features_metadata


def test_tabnet_classifier_works(setup_data):
    data, features_metadata = setup_data
    tabnet_classifier = TabNetClassifier(num_classes=2,
                                         features_metadata=features_metadata)
    tabnet_classifier(data)


def test_tabnet_regressor_works(setup_data):
    data, features_metadata = setup_data
    tabnet_regressor = TabNetRegressor(num_outputs=1,
                                       features_metadata=features_metadata)
    tabnet_regressor(data)
