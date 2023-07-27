import tensorflow as tf
from tensorflow import keras
from teras.utils.dnfnet import (tf_matmul_sparse_dense,
                                generate_random_mask,
                                extension_matrix,
                                binary_threshold)
from teras.utils.types import FloatSequence


@keras.saving.register_keras_serializable(package="teras.layers.dnfnet")
class DNFNetFeatureSelection(keras.layers.Layer):
    """
    DNFNetFeatureSelection layer based on the Feature Selection component
    proposed by Liran Katzir et al.
    in the paper Net-DNF: Effective Deep Modeling of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        data_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the input dataset.

        num_formulas: ``int``, default 256,
            Number of DNF formulas to use.
            Each DNF formula is analogous to a tree in tree based ensembles.

        keep_feature_prob_arr: ``List[float]`` or ``Tuple[float]``, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array.
            It is used  to randomly select a probability value that is used
            in the random selection of input features.

        elastic_net_beta: ``float``, default 0.4,
            Used in the computation of Elastic Net Regularization.

        binary_threshold_eps: ``float``, default 1.0,
            Used in the computation of learnable mask.
    """
    def __init__(self,
                 input_dim: int,
                 num_formulas: int = 256,
                 keep_feature_prob_arr: FloatSequence = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 **kwagrs):
        super().__init__(**kwagrs)
        self.input_dim = input_dim
        self.num_formulas = num_formulas
        self.keep_feature_prob_arr = keep_feature_prob_arr
        self.elastic_net_beta = elastic_net_beta
        self.binary_threshold_eps = binary_threshold_eps
        self.v_temp = tf.Variable(tf.zeros(self.num_formulas), trainable=False)
        self.elastic_net_alpha = None
        self.learnable_mask = None

    def build(self, input_shape):
        self.elastic_net_alpha = tf.Variable(initial_value=tf.constant(0.),
                                             name='elastic_net_alpha')
        self.learnable_mask = tf.Variable(initial_value=tf.fill(dims=(self.input_dim, self.num_formulas),
                                                                value=self.binary_threshold_eps + 0.5),
                                          shape=(self.input_dim, self.num_formulas),
                                          name='learnable_mask')

    def call(self,
             num_literals_per_formula_arr=None,
             ):
        literals_random_mask, formulas_random_mask = generate_random_mask(self.input_dim,
                                                                          self.keep_feature_prob_arr,
                                                                          num_literals_per_formula_arr,
                                                                          self.num_formulas)
        num_effective_features = tf.reduce_sum(formulas_random_mask, axis=0)
        formulas_random_mask = tf.reshape(formulas_random_mask, shape=tf.shape(self.learnable_mask))
        ext_matrix = extension_matrix(self.v_temp,
                                      self.num_formulas,
                                      num_literals_per_formula_arr)
        learnable_mask_01 = binary_threshold(self.learnable_mask, eps=self.binary_threshold_eps)

        l2_square_norm_selected = tf.linalg.diag_part(tf.matmul(tf.transpose(tf.square(self.learnable_mask)),
                                                                formulas_random_mask))
        l1_norm_selected = tf.linalg.diag_part(tf.matmul(tf.transpose(tf.abs(self.learnable_mask)),
                                                         formulas_random_mask))

        l2 = tf.abs(
            tf.divide(l2_square_norm_selected, num_effective_features) - self.elastic_net_beta * self.binary_threshold_eps ** 2)
        l1 = tf.abs(tf.divide(l1_norm_selected, num_effective_features) - self.elastic_net_beta * self.binary_threshold_eps)
        elastic_net_reg = tf.reduce_mean(
            (l2 * ((1 - tf.nn.sigmoid(self.elastic_net_alpha)) / 2) + l1 * tf.nn.sigmoid(self.elastic_net_alpha)))
        learnable_binary_mask = tf_matmul_sparse_dense(ext_matrix, learnable_mask_01)
        return learnable_binary_mask, literals_random_mask, elastic_net_reg

    def get_config(self):
        config = super().get_config()
        config.update({'input_dim': self.input_dim,
                       'num_formulas': self.num_formulas,
                       'keep_feature_prob_arr': self.keep_feature_prob_arr,
                       'elastic_net_beta': self.elastic_net_beta,
                       'binary_threshold_eps': self.binary_threshold_eps,
                       'v_temp': self.v_temp})
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim, **config)
