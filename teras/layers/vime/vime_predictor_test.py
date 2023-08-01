import tensorflow as tf
from teras.layers.vime.vime_predictor import VimePredictor


def test_vime_predictor_valid_call():
    inputs = tf.ones((8, 5))
    predictor = VimePredictor(num_labels=2,
                              hidden_dim=16)
    y_hat_logit, y_hat = predictor(inputs)


def test_vime_predictor_output_shape():
    inputs = tf.ones((8, 5))
    predictor = VimePredictor(num_labels=2,
                              hidden_dim=16)
    y_hat_logit, y_hat = predictor(inputs)

    assert len(tf.shape(y_hat_logit)) == 2
    assert tf.shape(y_hat_logit)[0] == 8
    assert tf.shape(y_hat_logit)[1] == 2

    assert len(tf.shape(y_hat)) == 2
    assert tf.shape(y_hat)[0] == 8
    assert tf.shape(y_hat)[1] == 2
