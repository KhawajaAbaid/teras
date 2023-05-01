from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from teras.layers import ObliviousDecisionTree
from teras.utils import sparsemoid
import tensorflow_addons as tfa


class NODERegressor(keras.Model):
    """
    Neural Oblivious Decision Tree (NODE) Regressor model
    based on the NODE architecture proposed by Sergei Popov et al.
    in paper Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        n_layers: Number of ObliviousDecisionTree layers to use in model
        max_features: Maximum number of features to use. If None, all features in the input dataset will be used.
        input_dropout: If None, no dropout will be applied to inputs.
            Otherwise, specified dropout rate will be applied to inputs.
        n_trees: Number of trees in ObliviousDecisionTree layer
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
                 n_layers=None,
                 max_features=None,
                 input_dropout=0.0,
                 n_trees=16,
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
        self.n_layers = n_layers
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.n_trees = n_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.choice_function = tfa.activations.sparsemax if choice_function is None else choice_function
        self.bin_function = sparsemoid if bin_function is None else bin_function
        self.response_initializer = response_initializer
        self.selection_logits_initializer = selection_logits_intializer
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff

        self.tree_layers = [ObliviousDecisionTree(n_trees=self.n_trees,
                                                  depth=self.depth,
                                                  tree_dim=self.tree_dim,
                                                  choice_function=self.choice_function,
                                                  bin_function=self.bin_function,
                                                  response_initializer=self.response_initializer,
                                                  selection_logits_intializer=self.selection_logits_initializer,
                                                  threshold_init_beta=self.threshold_init_beta,
                                                  threshold_init_cutoff=self.threshold_init_cutoff,
                                                  **kwargs)
                                                    for _ in range(n_layers)]
        if self.input_dropout:
            self.dropout = layers.Dropout(self.input_dropout)

    def call(self, inputs, **kwargs):
        x_out = inputs
        initial_features = inputs.shape[-1]
        for layer in self.tree_layers:
            x = x_out
            if self.max_features is not None:
                tail_features = min(self.max_features, x.shape[-1]) - initial_features
                if tail_features != 0:
                    x = tf.concat([x[..., :initial_features], x[..., -tail_features:]], axis=-1)
            if self.input_dropout:
                x = self.dropout(x)
            h = layer(x)
            x_out = tf.concat([x_out, h], axis=-1)
        outputs = x_out[..., initial_features:]
        return outputs


class NODEClassifier(keras.Model):
    """
    Neural Oblivious Decision Tree (NODE) Classifier model
    based on the NODE architecture proposed by Sergei Popov et al.
    in paper Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        n_layers: Number of ObliviousDecisionTree layers to use in model
        n_classes: Number of classes to predict
        activation_out: Actiavtion layer to use for output.
            By default 'sigmoid' is used for binary
            while 'softmax' is used for multiclass classification.
        max_features: Maximum number of features to use. If None, all features in the input dataset will be used.
        input_dropout: If None, no dropout will be applied to inputs.
            Otherwise, specified dropout rate will be applied to inputs.
        n_trees: Number of trees in ObliviousDecisionTree layer
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
                 n_layers=None,
                 n_classes=None,
                 activation_out=None,
                 max_features=None,
                 input_dropout=0.0,
                 n_trees=16,
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
        assert n_classes is not None, ("n_classes must not be None.")
        self.n_layers = n_layers
        self.n_classes = 1 if n_classes <= 2 else n_classes
        if activation_out is None:
            self.activation_out = "sigmoid" if self.n_classes <= 2 else 'softmax'
        else:
            self.activation_out = activation_out
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.n_trees = n_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.choice_function = tfa.activations.sparsemax if choice_function is None else choice_function
        self.bin_function = sparsemoid if bin_function is None else bin_function
        self.response_initializer = response_initializer
        self.selection_logits_initializer = selection_logits_intializer
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff

        self.tree_layers = [ObliviousDecisionTree(n_trees=self.n_trees,
                                                  depth=self.depth,
                                                  tree_dim=self.tree_dim,
                                                  choice_function=self.choice_function,
                                                  bin_function=self.bin_function,
                                                  response_initializer=self.response_initializer,
                                                  selection_logits_intializer=self.selection_logits_initializer,
                                                  threshold_init_beta=self.threshold_init_beta,
                                                  threshold_init_cutoff=self.threshold_init_cutoff,
                                                  **kwargs)
                            for _ in range(n_layers)]

        self.dense_out = layers.Dense(self.n_classes, activation=self.activation_out)

        if self.input_dropout:
            self.dropout = layers.Dropout(self.input_dropout)

    def call(self, inputs, **kwargs):
        x_out = inputs
        initial_features = inputs.shape[-1]
        for layer in self.tree_layers:
            x = x_out
            if self.max_features is not None:
                tail_features = min(self.max_features, x.shape[-1]) - initial_features
                if tail_features != 0:
                    x = tf.concat([x[..., :initial_features], x[..., -tail_features:]], axis=-1)
            if self.input_dropout:
                x = self.dropout(x)
            h = layer(x)
            x_out = tf.concat([x_out, h], axis=-1)
        outputs = x_out[..., initial_features:]
        outputs = self.dense_out(outputs)
        return outputs
