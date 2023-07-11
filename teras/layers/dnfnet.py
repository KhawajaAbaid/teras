import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from teras.utils.dnfnet import (compute_total_number_of_literals,
                                compute_total_number_of_conjunctions,
                                compute_num_literals_per_formula,
                                create_conjunctions_indicator_matrix,
                                create_formulas_indicator_matrix,
                                dense_ndim_array_to_sparse_tensor,
                                tf_matmul_sparse_dense,
                                and_operator,
                                or_operator,
                                generate_random_mask,
                                extension_matrix,
                                binary_threshold)
from teras.layers.common.head import (ClassificationHead as _BaseClassificationHead,
                                      RegressionHead as _BaseRegressionHead)
from typing import List, Union

LIST_OF_INT = List[int]
LIST_OF_FLOAT = List[float]
LIST_OR_TUPLE = Union[list, tuple]
LAYER_OR_STR = Union[keras.layers.Layer, str]


class FeatureSelection(keras.layers.Layer):
    """
    Feature Selection layer based on the Feature Selection component proposed by Liran Katzir et al.
    in the paper Net-DNF: Effective Deep Modeling of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_formulas: `int`, default 256,
            Number of DNF formulas to use.
            Each DNF formula is analogous to a tree in tree based ensembles.
        keep_feature_prob_arr: `List[float]`, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array.
            It is used  to randomly select a probability value that is used
            in the random selection of input features.
        elastic_net_beta: `float`, default 0.4,
            Used in the computation of Elastic Net Regularization.
        binary_threshold_eps:   `float`, default 1.0,
            Used in the computation of learnable mask.
    """
    def __init__(self,
                 input_dim: int,
                 num_formulas: int = 256,
                 keep_feature_prob_arr: LIST_OF_FLOAT = [0.1, 0.3, 0.5, 0.7, 0.9],
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
        new_config = {'input_dim': self.input_dim,
                      'num_formulas': self.num_formulas,
                      'keep_feature_prob_arr': self.keep_feature_prob_arr,
                      'elastic_net_beta': self.elastic_net_beta,
                      'binary_threshold_eps': self.binary_threshold_eps,
                      'v_temp': self.v_temp}
        config.update(new_config)
        return config


class Localization(keras.layers.Layer):
    """
    Localization layer based on the localization component proposed by Liran Katzir et al.
    in the paper Net-DNF: Effective Deep Modeling of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_formulas: `int`, default 256,
            Number of DNF formulas to use.
            Each DNF formula is analogous to a tree in tree based ensembles.
        temperature: `float`, default 2.0,
            Temperature value to use.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_formulas: int = 256,
                 temperature: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_formulas = num_formulas
        self.temperature = temperature
        self.softmax = layers.Softmax()

    def build(self, input_shape):
        input_dim = input_shape[1]
        mu_initializer = initializers.random_normal()
        self.mu = tf.Variable(initial_value=mu_initializer(shape=(self.num_formulas, input_dim)),
                              shape=(self.num_formulas, input_dim),
                              name='exp_mu')
        sigma_initializer = initializers.random_normal()
        self.sigma = tf.Variable(initial_value=sigma_initializer(shape=(1, self.num_formulas, input_dim)),
                                 shape=(1, self.num_formulas, input_dim),
                                 name="exp_sigma")
        self.temperature = tf.Variable(name='temperature',
                                       initial_value=tf.constant(value=self.temperature),
                                       dtype=tf.float32)

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.mu, axis=0)
        loc = tf.exp(-1 * tf.norm(tf.multiply(diff, self.sigma), axis=-1))
        loc = self.softmax(tf.sigmoid(self.temperature) * loc)
        return loc

    def get_config(self):
        config = super().get_config()
        new_config = {'num_formulas': self.num_formulas,
                      'temperature': self.temperature}
        config.update(new_config)
        return config


class DNNF(keras.layers.Layer):
    """
    Disjunctive Normal Neural Form (DNNF) layer
    is the  main building block of DNF-Net architecture.
    Based on the paper Net-DNF: Effective Deep Modeling of Tabular Data by Liran Katzir et al.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_formulas: `int`, default 256,
            Number of DNF formulas. Each DNF formula is analogous to a tree in tree based ensembles.
        num_conjunctions_arr: `List[int]`, default [6, 9, 12, 15],
            Conjunctions array.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        conjunctions_depth_arr: `List[int]`, default [2, 4, 6],
            Conjunctions depth array.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        keep_feature_prob_arr: `List[float]`, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.
        elastic_net_beta: `float`, default 0.4,
            Used in the computation of Elastic Net Regularization.
        binary_threshold_eps:   `float`, default 1.0,
            Used in the computation of learnable mask.
        temperature: `float`, default 2.0,
            Temperature value to use in the Localization layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_formulas: int = 256,
                 num_conjunctions_arr: LIST_OF_INT = [6, 9, 12, 15],
                 conjunctions_depth_arr: LIST_OF_INT = [2, 4, 6],
                 keep_feature_prob_arr: LIST_OF_FLOAT = [0.1, 0.3, 0.5, 0.7, 0.9],
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

        self.localization = Localization(num_formulas=self.num_formulas,
                                         temperature=self.temperature)

        self.loaded_input_masks = None

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.feature_selection = FeatureSelection(input_dim=input_dim,
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
                                      initializer=initializers.glorot_normal())
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
        new_config = {'num_formulas': self.num_formulas,
                      'num_conjunctions_arr': self.num_conjunctions_arr,
                      'conjunctions_depth_arr': self.conjunctions_depth_arr,
                      'keep_feature_prob_arr': self.keep_feature_prob_arr,
                      'elastic_net_beta': self.elastic_net_beta,
                      'binary_threshold_eps': self.binary_threshold_eps,
                      'temperature': self.temperature}
        config.update(new_config)
        return config


class ClassificationHead(_BaseClassificationHead):
    """
    Classification head for DNFNet Classifier model.

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        units_values: `List[int] | Tuple[int]`, default `None`,
            If specified, for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the ClassificationHead.
        activation_hidden: default `None`,
            Activation function to use in hidden dense layers.
        activation_out: default `None`,
            Activation function to use for the output layer.
            If not specified, `sigmoid` is used for binary and `softmax` is used for
            multiclass classification.
        normalization: `Layer | str`, default `None`,
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_classes: int = 2,
                 units_values: LIST_OR_TUPLE = None,
                 activation_hidden=None,
                 activation_out=None,
                 normalization: LAYER_OR_STR = None,
                 **kwargs):
        super().__init__(num_classes=num_classes,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         activation_out=activation_out,
                         normalization=normalization,
                         **kwargs)



class RegressionHead(_BaseRegressionHead):
    """
    Regression head for the DNFNet Regressor model.

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs to predict.
        units_values: `List[int] | Tuple[int]`, default `None`,
            If specified, for each value in the sequence
            a hidden layer of that dimension preceded by a normalization layer (if specified) is
            added to the RegressionHead.
        activation_hidden: default `None`,
            Activation function to use in hidden dense layers.
        normalization: `Layer | str`, default `None`,
            Normalization layer to use.
            If specified a normalization layer is applied after each hidden layer.
            If None, no normalization layer is applied.
            You can either pass a keras normalization layer or name for a layer implemented by keras.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 units_values: LIST_OR_TUPLE = None,
                 activation_hidden=None,
                 normalization: LAYER_OR_STR = None,
                 **kwargs):
        super().__init__(num_outputs=num_outputs,
                         units_values=units_values,
                         activation_hidden=activation_hidden,
                         normalization=normalization,
                         **kwargs)
