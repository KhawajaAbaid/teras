import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
from teras.utils.DNFNet import (compute_total_number_of_literals,
                                compute_total_number_of_conjunctions,
                                compute_n_literals_per_formula,
                                create_conjunctions_indicator_matrix,
                                create_formulas_indicator_matrix,
                                dense_ndim_array_to_sparse_tensor,
                                feature_selection,
                                tf_matmul_sparse_dense,
                                and_operator,
                                or_operator,
                                broadcast_exp,
                                generate_random_mask,
                                extension_matrix,
                                binary_threshold)



class FeatureSelection(keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 n_formulas=None,
                 elastic_net_beta=None,
                 binary_threshold_eps=1,
                 **kwagrs):
        super().__init__(**kwagrs)
        self.input_dim = input_dim
        # self.keep_feature_prob_arr = keep_feature_prob_arr
        # self.n_literals_per_formula_arr = n_literals_per_formula_arr
        # self.learnable_mask = learnable_mask
        self.n_formulas = n_formulas
        # self.elastic_net_alpha = elastic_net_alpha
        self.elastic_net_beta = elastic_net_beta
        # self.extension_matrix_v = extension_matrix_v
        self.binary_threshold_eps = binary_threshold_eps
        self.v_temp = tf.Variable(tf.zeros(self.n_formulas), trainable=False)

    def build(self, input_shape):
        self.elastic_net_alpha = tf.Variable(initial_value=tf.constant(0.),
                                             name='elastic_net_alpha')
        self.learnable_mask = tf.Variable(initial_value=tf.fill(dims=(self.input_dim, self.n_formulas),
                                                                value=self.binary_threshold_eps + 0.5),
                                          shape=(self.input_dim, self.n_formulas),
                                          name='learnable_mask')
    def call(self,
             keep_feature_prob_arr=None,
             n_literals_per_formula_arr=None,
             ):
        literals_random_mask, formulas_random_mask = generate_random_mask(self.input_dim,
                                                                          keep_feature_prob_arr,
                                                                          n_literals_per_formula_arr,
                                                                          self.n_formulas)
        n_effective_features = tf.reduce_sum(formulas_random_mask, axis=0)
        formulas_random_mask = tf.reshape(formulas_random_mask, shape=tf.shape(self.learnable_mask))
        ext_matrix = extension_matrix(self.v_temp,
                                      self.n_formulas,
                                      n_literals_per_formula_arr)
        learnable_mask_01 = binary_threshold(self.learnable_mask, eps=self.binary_threshold_eps)

        l2_square_norm_selected = tf.linalg.diag_part(tf.matmul(tf.transpose(tf.square(self.learnable_mask)),
                                                                formulas_random_mask))
        l1_norm_selected = tf.linalg.diag_part(tf.matmul(tf.transpose(tf.abs(self.learnable_mask)),
                                                         formulas_random_mask))

        l2 = tf.abs(
            tf.divide(l2_square_norm_selected, n_effective_features) - self.elastic_net_beta * self.binary_threshold_eps ** 2)
        l1 = tf.abs(tf.divide(l1_norm_selected, n_effective_features) - self.elastic_net_beta * self.binary_threshold_eps)
        elastic_net_reg = tf.reduce_mean(
            (l2 * ((1 - tf.nn.sigmoid(self.elastic_net_alpha)) / 2) + l1 * tf.nn.sigmoid(self.elastic_net_alpha)))
        learnable_binary_mask = tf_matmul_sparse_dense(ext_matrix, learnable_mask_01)
        return learnable_binary_mask, literals_random_mask, elastic_net_reg



class Localization(keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 n_formulas,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_formulas = n_formulas
        self.softmax = layers.Softmax()

    def build(self, input_shape):
        mu_initializer = initializers.random_normal()
        self.mu = tf.Variable(initial_value=mu_initializer(shape=(self.n_formulas, self.input_dim)),
                         shape=(self.n_formulas, self.input_dim), name='exp_mu')
        sigma_initializer = initializers.random_normal()
        self.sigma = tf.Variable(initial_value=sigma_initializer(shape=(1, self.n_formulas, self.input_dim)),
                            shape=(1, self.n_formulas, self.input_dim), name="exp_sigma")
        self.temperature = tf.Variable(name='temperature',
                                       initial_value=tf.constant(value=2.))

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.mu, axis=0)
        loc = tf.exp(-1 * tf.norm(tf.multiply(diff, self.sigma), axis=-1))
        loc = self.softmax(tf.sigmoid(self.temperature) * loc)
        return loc

class DNNF(keras.layers.Layer):
    """
    Disjunctive Normal Neural Form (DNNF) layer
    is the  main building block of DNF-Net architecture.
    Based on the paper Net-DNF: Effective Deep Modeling of Tabular Data by Liran Katzir et al.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        n_conjunctions_arr: Conjunctions array. If None, default values will be used.
        conjunctions_depth_arr: Conjunctions depth array. If None, default values will be used.
        keep_feature_prob_arr: Feature probability array. If None. default values will be used.
        n_formulas: Number of formulas to use in DNNF layer
        elastic_net_beta: Use in the creation of Elastic Net Reg. Defaults to 0.4
        binary_threshold_eps: Used in the creation of learnable mask. Defaults to 1
    """
    def __init__(self,
                 n_conjunctions_arr=None,
                 conjunctions_depth_arr=None,
                 keep_feature_prob_arr=None,
                 n_formulas=256,
                 elastic_net_beta=0.4,
                 binary_threshold_eps=1):
        super().__init__()
        self.loaded_input_masks = None
        self.n_conjunctions_arr = [6, 9, 12, 15] if n_conjunctions_arr is None else n_conjunctions_arr
        self.conjunctions_depth_arr = [2, 4, 6] if conjunctions_depth_arr is None else conjunctions_depth_arr
        self.keep_feature_prob_arr = [0.1, 0.3, 0.5, 0.7, 0.9] if keep_feature_prob_arr is None else keep_feature_prob_arr
        self.n_formulas = n_formulas
        self.elastic_net_beta = elastic_net_beta
        self.binary_threshold_eps = binary_threshold_eps
        # self.softmax = layers.Softmax()


    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        self.feature_selection = FeatureSelection(input_dim=self.input_dim,
                                                  n_formulas=self.n_formulas,
                                                  elastic_net_beta=self.elastic_net_beta
                                                  )
        self.localization = Localization(input_dim=self.input_dim,
                                         n_formulas=self.n_formulas)

        # mu_initializer = initializers.random_normal()
        # self.mu = tf.Variable(initial_value=mu_initializer(shape=(self.n_formulas, self.input_dim)),
        #                  shape=(self.n_formulas, self.input_dim), name='exp_mu')
        # sigma_initializer = initializers.random_normal()
        # self.sigma = tf.Variable(initial_value=sigma_initializer(shape=(1, self.n_formulas, self.input_dim)),
        #                     shape=(1, self.n_formulas, self.input_dim), name="exp_sigma")

        # Learnable variables for feature selection
        # self.elastic_net_alpha = tf.Variable(initial_value=tf.constant(0.),
        #                                      name='elastic_net_alpha')
        # self.learnable_mask = tf.Variable(initial_value=tf.fill(dims=(self.input_dim, self.n_formulas),
        #                                                         value=self.binary_threshold_eps + 0.5),
        #                                   shape=(self.input_dim, self.n_formulas),
        #                                   name='learnable_mask')

        self.total_number_of_literals = compute_total_number_of_literals(self.n_formulas, self.n_conjunctions_arr,
                                                                    self.conjunctions_depth_arr)
        n_literals_per_formula_arr = compute_n_literals_per_formula(self.n_conjunctions_arr,
                                                                    self.conjunctions_depth_arr)
        total_number_of_conjunctions = compute_total_number_of_conjunctions(self.n_formulas, self.n_conjunctions_arr)

        # extension_matrix_v is a prop to hold intermediate values.
        # it isn't a matrix itself, but it's a build block for extension matrix
        # The reason for using this variable is, because we can't assign values to a tensor.
        # extension_matrix_v = tf.Variable(tf.zeros(self.n_formulas), trainable=False)
        # self.learnable_binary_mask, self.literals_random_mask, self.elastic_net_reg = feature_selection(self.input_dim,
        #                                                                                 self.keep_feature_prob_arr,
        #                                                                                 n_literals_per_formula_arr,
        #                                                                                 self.n_formulas,
        #                                                                                 self.elastic_net_beta,
        #                                                                                 self.learnable_mask,
        #                                                                                 self.elastic_net_alpha,
        #                                                                                 extension_matrix_v)
        self.learnable_binary_mask, self.literals_random_mask, self.elastic_net_reg = self.feature_selection(
                                                                                        self.keep_feature_prob_arr,
                                                                                        n_literals_per_formula_arr)

        # tf.shape(self.literals_random_mask)[1] == total_number_of_literals
        self.bias = tf.Variable(name='literals_bias',
                                shape=tf.shape(self.literals_random_mask)[1],
                                initial_value=tf.fill(dims=(tf.shape(self.literals_random_mask)[1],), value=0.))
        self.weight = self.add_weight(name='literals_weights',
                                      shape=tf.shape(self.literals_random_mask),
                                      initializer=initializers.glorot_normal())
        self.weight = tf.multiply(self.weight, self.literals_random_mask)
        # self.temperature = tf.Variable(name='temperature',
        #                                initial_value=tf.constant(value=2.))
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
        self.conjunctions_indicator_sparse_matrix = dense_ndim_array_to_sparse_tensor(conjunctions_indicator_matrix)
        self.and_bias = np.sum(conjunctions_indicator_matrix,
                               axis=0, dtype=np.float32)

        c = tf.Variable(tf.cast(tf.zeros(total_number_of_conjunctions), dtype=tf.bool))
        formulas_indicator_matrix = create_formulas_indicator_matrix(c,
                                                                     self.n_formulas,
                                                                     self.n_conjunctions_arr)
        self.formulas_indicator_sparse_matrix = dense_ndim_array_to_sparse_tensor(formulas_indicator_matrix)
        self.or_bias = np.sum(formulas_indicator_matrix, axis=0, dtype=np.float32)

    def call(self, inputs):
        x = inputs
        # Creating affine literals
        out_literals = tf.tanh(tf.add(tf.matmul(x, tf.multiply(self.weight, self.learnable_binary_mask)), self.bias))
        out_conjunctions = and_operator(tf_matmul_sparse_dense(self.conjunctions_indicator_sparse_matrix, out_literals),
                                        d=self.and_bias)
        out_DNNFs = or_operator(tf_matmul_sparse_dense(self.formulas_indicator_sparse_matrix, out_conjunctions),
                                d=self.or_bias)
        out_DNNFs = tf.reshape(out_DNNFs, shape=(-1, self.n_formulas))
        # loc = broadcast_exp(x, mu=self.mu, sigma=self.sigma)
        # loc = self.softmax(tf.sigmoid(self.temperature) * loc)
        loc = self.localization(x)
        out_DNNFs = tf.multiply(out_DNNFs, loc)
        return out_DNNFs

