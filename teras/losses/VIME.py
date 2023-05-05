import tensorflow as tf
from tensorflow import keras


class VimeSelfSupervisedLoss(keras.losses.Loss):
    """Self supervised loss as proposed by Jinsung Yoon et al.
    in the paper "VIME: Extending the Success of Self- and
    Semi-supervised Learning to Tabular Domain"

    Reference(s):
        https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html
    """
    def call(self, y_true=None, y_pred=None):
        """
        Args:
            y_pred: yv_hat_logit
        """
        yu_loss = tf.reduce_mean(tf.nn.moments(y_pred, axes=0)[1])
        return yu_loss