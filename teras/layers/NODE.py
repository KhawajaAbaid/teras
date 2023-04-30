from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from warnings import warn
from teras.utils import sparsemoid
import tensorflow_addons as tfa
import tensorflow_probability as tfp


class ObliviousDecisionTree(layers.Layer):
    """
        Oblivious Decision Tree layer as proposed by Sergei Popov et al.
        in paper Neural Oblivious Decision Ensembles
        for Deep Learning on Tabular Data

        Reference(s):
            https://arxiv.org/abs/1909.06312

        Args:
            n_trees: Number of trees in this layer
            depth: Number of splits in every tree
            tree_dim: Number of response channels in the response of individual tree
            choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
            bin_function: f(tensor) -> R[0, 1], computes tree leaf weights
            response_initializer: Initializer for tree output tensor
            selection_logits_intializer: Initializer for logits that select features for the tree
            Both thresholds and scales are initialized with data-aware initialization function.
            threshold_init_beta: initializes threshold to a q-th quantile of data points
                where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
                If this param is set to 1, initial thresholds will have the same distribution as data points
                If greater than 1 (e.g. 10), thresholds will be closer to median data value
                If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

            threshold_init_cutoff: threshold log-temperatures initializer, \in (0, inf)
                By default(1.0), log-remperatures are initialized in such a way that all bin selectors
                end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
                Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
                Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
                For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
                Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
                All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
    """
    def __init__(self,
                 n_trees,
                 depth=6,
                 tree_dim=1,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_intializer="random_uniform",
                 threshold_init_beta=1.0,
                 threshold_init_cutoff=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_trees = n_trees
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
                                        shape=(self.n_trees, self.tree_dim, 2 ** self.depth))
        self.feature_selection_logits = self.add_weight(initializer=self.selection_logits_initializer,
                                                        shape=(input_dim, self.n_trees, self.depth))
        self.feature_thresholds = tf.Variable(initial_value=keras.initializers.zeros()(
                                                                                shape=(self.n_trees, self.depth),
                                                                                dtype="float32"),
                                              shape=[self.n_trees, self.depth])

        self.log_temperatures = tf.Variable(initial_value=keras.initializers.zeros()(
                                                                                shape=(self.n_trees, self.depth),
                                                                                dtype="float32"),
                                            shape=[self.n_trees, self.depth])

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
        percentiles_q = 100 * beta_dist.sample([self.n_trees * self.depth])

        flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, feature_values)

        feature_thresholds = tfp.stats.percentile(flattened_feature_values, percentiles_q)
        feature_thresholds = tf.reshape(feature_thresholds, shape=(self.n_trees, self.depth))
        temperatures = tfp.stats.percentile(tf.abs(feature_values - feature_thresholds),
                                     q=100 * min(1.0, self.threshold_init_cutoff), axis=0)

        # if threshold_init_cutoff > 1, scale everything down by it
        temperatures /= max(1.0, self.threshold_init_cutoff)
        log_tempratures = tf.math.log(temperatures + eps)
        log_tempratures = tf.reshape(log_tempratures, shape=[self.n_trees, self.depth])

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
