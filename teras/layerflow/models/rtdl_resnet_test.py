from teras.layerflow.models.rtdl_resnet import RTDLResNetClassifier, RTDLResNetRegressor
import tensorflow as tf


def test_rtdl_resnet_classifier_works():
    data = tf.ones(shape=(16, 10))
    rtdl_resnet_classifier = RTDLResNetClassifier()
    rtdl_resnet_classifier(data)


def test_rtdl_resnet_regressor_works():
    data = tf.ones(shape=(16, 10))
    rtdl_resnet_regressor = RTDLResNetRegressor()
    rtdl_resnet_regressor(data)
