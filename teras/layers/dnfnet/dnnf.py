import tensorflow as tf
from tensorflow import keras
from teras.utils.dnfnet import (compute_total_number_of_literals,
                                compute_total_number_of_conjunctions,
                                compute_num_literals_per_formula,
                                create_conjunctions_indicator_matrix,
                                create_formulas_indicator_matrix,
                                dense_ndim_array_to_sparse_tensor,
                                tf_matmul_sparse_dense,
                                and_operator,
                                or_operator)
from teras.layers.dnfnet.dnfnet_localization import DNFNetLocalization
from teras.layers.dnfnet.dnfnet_feature_selection import DNFNetFeatureSelection
from teras.utils.types import (IntegerSequence,
                               FloatSequence)


@keras.saving.register_keras_serializable(package="teras.layers.dnfnet")
class DNNF(keras.layers.Layer):
    """
    Disjunctive Normal Neural Form (DNNF) layer
    is the  main building block of DNF-Net architecture.
    Based on the paper Net-DNF: Effective Deep Modeling of Tabular Data by Liran Katzir et al.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_formulas: ``int``, default 256,
            Number of DNF formulas. Each DNF formula is analogous to a tree in tree based ensembles.

        num_conjunctions_arr: ``List[int]`` or ``Tuple[int]``, default [6, 9, 12, 15],
            Conjunctions array.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        conjunctions_depth_arr: ``List[int]`` or ``Tuple[int]``, default [2, 4, 6],
            Conjunctions depth array.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        keep_feature_prob_arr: ``List[float]`` or ``Tuple[float]``, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.

        elastic_net_beta: ``float``, default 0.4,
            Used in the computation of Elastic Net Regularization.

        binary_threshold_eps:   ``float``, default 1.0,
            Used in the computation of learnable mask.

        temperature: ``float``, default 2.0,
            Temperature value to use in the Localization layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """

    def __init__(self,
                 num_formulas: int = 256,
                 num_conjunctions_arr: IntegerSequence = [6, 9, 12, 15],
                 conjunctions_depth_arr: IntegerSequence = [2, 4, 6],
                 keep_feature_prob_arr: FloatSequence = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 temperature: float = 2.0,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.num_formulas = num_formulas
        self.num_conjunctions_arr = num_conjunctions_arr
        self.conjunctions_depth_arr = conjunctions_depth_arr
        self.keep_feature_prob_arr = keep_feature_prob_arr
        self.elastic_net_beta = elastic_net_beta
        self.binary_threshold_eps = binary_threshold_eps
        self.temperature = temperature

        self.localization = DNFNetLocalization(num_formulas=self.num_formulas,
                                               temperature=self.temperature)

        self.loaded_input_masks = None
        self.feature_selection = None

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.feature_selection = DNFNetFeatureSelection(input_dim=input_dim,
                                                        num_formulas=self.num_formulas,
                                                        keep_feature_prob_arr=self.keep_feature_prob_arr,
                                                        elastic_net_beta=self.elastic_net_beta,
                                                        binary_threshold_eps=self.binary_threshold_eps)

        self.total_number_of_literals = compute_total_number_of_literals(self.num_formulas,
                                                                         self.num_conjunctions_arr,
                                                                         self.conjunctions_depth_arr)
        num_literals_per_formula_arr = compute_num_literals_per_formula(self.num_conjunctions_arr,
                                                                        self.conjunctions_depth_arr)
        total_number_of_conjunctions = compute_total_number_of_conjunctions(self.num_formulas,
                                                                            self.num_conjunctions_arr)

        self.learnable_binary_mask, self.literals_random_mask, self.elastic_net_reg = self.feature_selection(
            num_literals_per_formula_arr)

        self.bias = tf.Variable(name='literals_bias',
                                shape=tf.shape(self.literals_random_mask)[1],
                                initial_value=tf.fill(dims=(tf.shape(self.literals_random_mask)[1],), value=0.))
        self.weight = self.add_weight(name='literals_weights',
                                      shape=tf.shape(self.literals_random_mask),
                                      initializer=keras.initializers.glorot_normal())
        self.weight = tf.multiply(self.weight, self.literals_random_mask)

        # 'b' is just a prop to hold intermediate values since we can't assign values to a tensor,
        # so we define a variable here with trainable argument set to False.
        # same is the case with 'c' below.
        b = tf.Variable(tf.cast(tf.zeros(tf.shape(self.literals_random_mask)[1]),
                                dtype=tf.bool),
                        trainable=False)
        conjunctions_indicator_matrix = create_conjunctions_indicator_matrix(b,
                                                                             total_number_of_conjunctions,
                                                                             self.conjunctions_depth_arr,
                                                                             )
        conjunctions_indicator_matrix = tf.cast(conjunctions_indicator_matrix, dtype=tf.float32)
        self.conjunctions_indicator_sparse_matrix = dense_ndim_array_to_sparse_tensor(conjunctions_indicator_matrix)
        self.and_bias = tf.reduce_sum(conjunctions_indicator_matrix,
                                      axis=0)

        c = tf.Variable(tf.cast(tf.zeros(total_number_of_conjunctions),
                                dtype=tf.bool))
        formulas_indicator_matrix = create_formulas_indicator_matrix(c,
                                                                     self.num_formulas,
                                                                     self.num_conjunctions_arr)
        formulas_indicator_matrix = tf.cast(formulas_indicator_matrix, dtype=tf.float32)
        self.formulas_indicator_sparse_matrix = dense_ndim_array_to_sparse_tensor(formulas_indicator_matrix)
        self.or_bias = tf.reduce_sum(formulas_indicator_matrix,
                                     axis=0)

    def call(self, inputs):
        x = inputs
        # Creating affine literals
        out_literals = tf.tanh(tf.add(tf.matmul(x, tf.multiply(self.weight, self.learnable_binary_mask)), self.bias))

        # Soft Conjunctions
        out_conjunctions = and_operator(tf_matmul_sparse_dense(self.conjunctions_indicator_sparse_matrix, out_literals),
                                        d=self.and_bias)
        out_DNNFs = or_operator(tf_matmul_sparse_dense(self.formulas_indicator_sparse_matrix, out_conjunctions),
                                d=self.or_bias)
        out_DNNFs = tf.reshape(out_DNNFs, shape=(-1, self.num_formulas))

        # Localization
        loc = self.localization(x)
        out_DNNFs = tf.multiply(out_DNNFs, loc)
        return out_DNNFs

    def get_config(self):
        config = super().get_config()
        config.update({'num_formulas': self.num_formulas,
                       'num_conjunctions_arr': self.num_conjunctions_arr,
                       'conjunctions_depth_arr': self.conjunctions_depth_arr,
                       'keep_feature_prob_arr': self.keep_feature_prob_arr,
                       'elastic_net_beta': self.elastic_net_beta,
                       'binary_threshold_eps': self.binary_threshold_eps,
                       'temperature': self.temperature}
                      )
        return config
