from teras.layerflow.models.dnfnet import DNFNetClassifier, DNFNetRegressor
import tensorflow as tf


def test_rtdl_resnet_classifier_works():
    data = tf.ones(shape=(16, 10))
    dnfnet_classifier = DNFNetClassifier(num_formulas=16)
    dnfnet_classifier(data)


def test_rtdl_resnet_regressor_works():
    data = tf.ones(shape=(16, 10))
    dnfnet_regressor = DNFNetRegressor(num_formulas=16)
    dnfnet_regressor(data)
