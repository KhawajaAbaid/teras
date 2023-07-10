from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from warnings import warn
from teras.utils import sparsemoid
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from teras.layers.common.head import (ClassificationHead as _BaseClassificationHead,
                                      RegressionHead as _BaseRegressionHead)
from typing import List, Union

LIST_OR_TUPLE = Union[list, tuple]
LAYER_OR_STR = Union[keras.layers.Layer, str]


class ObliviousDecisionTree(layers.Layer):
    """
        Oblivious Decision Tree layer as proposed by Sergei Popov et al.
        in paper Neural Oblivious Decision Ensembles
        for Deep Learning on Tabular Data

        Reference(s):
            https://arxiv.org/abs/1909.06312

        Args:
       num_trees: `int`, default 128,
            Number of trees
        depth: `int`, default 6,
            Number of splits in every tree
        tree_dim: `int`, default 1,
            Number of response channels in the response of individual tree
        max_features: `int`,
            Maximum number of features to use. If None, all features in the input dataset will be used.
        input_dropout: `float`, default 0.,
            Dropout rate to apply to inputs.
        choice_function:
            f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
            By default, sparsemax is used.
        bin_function:
            f(tensor) -> R[0, 1], computes tree leaf weights
            By default, sparsemoid is used.
        response_initializer: default "random_normal",
            Initializer for tree output tensor. Any format that is acceptable by the keras initializers.
        selection_logits_intializer: default "random_uniform",
            Initializer for logits that select features for the tree
            Both thresholds and scales are initialized with data-aware initialization function.
        threshold_init_beta: `float`, default 1.0,
            Initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.
        threshold_init_cutoff: `float`, default 1.0,
            Threshold log-temperatures initializer, \in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
    """
    def __init__(self,
                 num_trees: int = 128,
                 depth: int = 6,
                 tree_dim: int = 1,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_intializer="random_uniform",
                 threshold_init_beta: float = 1.0,
                 threshold_init_cutoff: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.choice_function = tfa.activations.sparsemax if choice_function is None else choice_function
        self.bin_function = sparsemoid if bin_function is None else bin_function
        self.response_initializer = response_initializer
        self.selection_logits_initializer = selection_logits_intializer
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff
        self.initialized = False

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.response = self.add_weight(initializer=self.response_initializer,
                                        shape=(self.num_trees, self.tree_dim, 2 ** self.depth))
        self.feature_selection_logits = self.add_weight(initializer=self.selection_logits_initializer,
                                                        shape=(input_dim, self.num_trees, self.depth))
        self.feature_thresholds = tf.Variable(initial_value=keras.initializers.zeros()(
                                                                                shape=(self.num_trees, self.depth),
                                                                                dtype="float32"),
                                              shape=[self.num_trees, self.depth])

        self.log_temperatures = tf.Variable(initial_value=keras.initializers.zeros()(
                                                                                shape=(self.num_trees, self.depth),
                                                                                dtype="float32"),
                                            shape=[self.num_trees, self.depth])

        indices = keras.backend.arange(2 ** self.depth)
        offsets = 2 ** keras.backend.arange(self.depth)
        bin_codes = tf.reshape(indices, shape=(1, -1)) // tf.reshape(offsets, shape=(-1, 1)) % 2
        bin_codes = tf.cast(bin_codes, dtype="float32")
        bin_codes_1hot = tf.stack([bin_codes, 1.0 - bin_codes], axis=-1)
        self.bin_codes_1hot = tf.Variable(initial_value=bin_codes_1hot, trainable=False)

    def _data_aware_initialization(self, inputs, eps=1e-6):
        """
        Data aware initialization as proposed by the paper authors.
        As the name implies, the function initializes variables based on the inputs.
        Hence, it is called when the layer's call method is called on some inputs
        and not in __init__ or build methods.
        """
        assert len(inputs.shape) == 2
        if tf.shape(inputs)[0] < 1000:
            warn("Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                 "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                 "You can do so manually before training.")
        feature_selectors = self.choice_function(self.feature_selection_logits, axis=0)
        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        beta_dist = tfp.distributions.Beta(self.threshold_init_beta, self.threshold_init_beta)
        percentiles_q = 100 * beta_dist.sample([self.num_trees * self.depth])

        flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, feature_values)

        feature_thresholds = tfp.stats.percentile(flattened_feature_values, percentiles_q)
        feature_thresholds = tf.reshape(feature_thresholds, shape=(self.num_trees, self.depth))
        temperatures = tfp.stats.percentile(tf.abs(feature_values - feature_thresholds),
                                     q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

        # if threshold_init_cutoff > 1, scale everything down by it
        temperatures /= max(1.0, self.threshold_init_cutoff)
        log_tempratures = tf.math.log(temperatures + eps)
        log_tempratures = tf.reshape(log_tempratures, shape=[self.num_trees, self.depth])

        self.feature_thresholds.assign(feature_thresholds)
        self.log_temperatures.assign(log_tempratures)

    def call(self, inputs):
        if len(inputs.shape) > 2:
            return self.call(tf.reshape(tf.reshape(inputs, shape=(-1, inputs.shape[-1])), shape=(*inputs.shape[:-1], -1)))
        if not self.initialized:
            self._data_aware_initialization(inputs)
            self.initialized = True

        feature_logits = self.feature_selection_logits
        features_selectors = self.choice_function(feature_logits, axis=0)
        feature_values = tf.einsum('bi,ind->bnd', inputs, features_selectors)
        threshold_logits = (feature_values - self.feature_thresholds) * tf.exp(-self.log_temperatures)
        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)
        bins = self.bin_function(threshold_logits)
        bin_matches = tf.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        response_weights = tf.reduce_prod(bin_matches, axis=-2)
        response = tf.einsum('bnd,ncd->bnc', response_weights, self.response)
        return tf.map_fn(keras.backend.flatten, response)

    def get_config(self):
        config = super().get_config()
        new_config = {'num_trees': self.num_trees,
                      'depth': self.depth,
                      'tree_dim': self.tree_dim,
                      'choice_function': self.choice_function,
                      'bin_function': self.bin_function,
                      'response_initializer': self.response_initializer,
                      'selection_logits_intializer': self.selection_logits_intializer,
                      'threshold_init_beta': self.threshold_init_beta,
                      'threshold_init_cutoff': self.threshold_init_cutoff,
                      }

        config.update(new_config)
        return config


class ClassificationHead(_BaseClassificationHead):
    """
    Classification head for NODE Classifier model.

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
    Regression head for the NODE Regressor model.

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
