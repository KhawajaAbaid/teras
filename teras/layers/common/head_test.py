from teras.layers.common.head import ClassificationHead, RegressionHead
import tensorflow as tf


def test_classification_head_binary_classification_output_shape():
    head = ClassificationHead(num_classes=2)
    inputs = tf.ones(shape=(128, 16), dtype=tf.float32)
    outputs = head(inputs)
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 1


def test_classification_head_multiclass_classification_output_shape():
    head = ClassificationHead(num_classes=3)
    inputs = tf.ones(shape=(128, 16), dtype=tf.float32)
    outputs = head(inputs)
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 3


def test_regression_head_single_regression_output_shape():
    head = RegressionHead(num_outputs=1)
    inputs = tf.ones(shape=(128, 16), dtype=tf.float32)
    outputs = head(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 1


def test_regression_head_multi_regression_output_shape():
    head = RegressionHead(num_outputs=3)
    inputs = tf.ones(shape=(128, 16), dtype=tf.float32)
    outputs = head(inputs)
    assert len(tf.shape(outputs)) == 2
    assert tf.shape(outputs)[0] == 128
    assert tf.shape(outputs)[1] == 3


