from teras.layerflow.models.tabtransformer import TabTransformerClassifier, TabTransformerRegressor
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


def test_tabtransformer_classifier_works(setup_data):
    data, features_metadata = setup_data
    tabtransformer_classifier = TabTransformerClassifier(num_classes=2,
                                                         features_metadata=features_metadata)
    tabtransformer_classifier(data)


def test_tabtransformer_regressor_works(setup_data):
    data, features_metadata = setup_data
    tabtransformer_regressor = TabTransformerRegressor(num_outputs=1,
                                                       features_metadata=features_metadata)
    tabtransformer_regressor(data)
