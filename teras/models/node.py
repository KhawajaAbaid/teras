from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from teras.layers import ObliviousDecisionTree
from teras.utils import sparsemoid
import tensorflow_addons as tfa


class NODE(keras.Model):
    """
    Neural Oblivious Decision Tree (NODE) model
    based on the NODE architecture proposed by Sergei Popov et al.
    in paper Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        num_layers: `int`, default 8,
            Number of ObliviousDecisionTree layers to use in model
        num_trees: `int`, default 128,
            Number of trees to use in each `ObliviousDecisionTree` layer
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
                 num_layers: int = 8,
                 num_trees: int = 16,
                 depth: int = 6,
                 tree_dim: int = 1,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_intializer="random_uniform",
                 threshold_init_beta: float = 1.0,
                 threshold_init_cutoff: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.choice_function = tfa.activations.sparsemax if choice_function is None else choice_function
        self.bin_function = sparsemoid if bin_function is None else bin_function
        self.response_initializer = response_initializer
        self.selection_logits_initializer = selection_logits_intializer
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff

        self.tree_layers = [ObliviousDecisionTree(num_trees=self.num_trees,
                                                  depth=self.depth,
                                                  tree_dim=self.tree_dim,
                                                  choice_function=self.choice_function,
                                                  bin_function=self.bin_function,
                                                  response_initializer=self.response_initializer,
                                                  selection_logits_intializer=self.selection_logits_initializer,
                                                  threshold_init_beta=self.threshold_init_beta,
                                                  threshold_init_cutoff=self.threshold_init_cutoff,
                                                  **kwargs)
                                                    for _ in range(num_layers)]
        self.dropout = layers.Dropout(self.input_dropout)
        self.head = None

    def call(self, inputs, **kwargs):
        x_out = inputs
        initial_features = inputs.shape[-1]
        for layer in self.tree_layers:
            x = x_out
            if self.max_features is not None:
                tail_features = min(self.max_features, x.shape[-1]) - initial_features
                if tail_features != 0:
                    x = tf.concat([x[..., :initial_features], x[..., -tail_features:]], axis=-1)
            x = self.dropout(x)
            h = layer(x)
            x_out = tf.concat([x_out, h], axis=-1)
        outputs = x_out[..., initial_features:]
        if self.head is not None:
            outputs = self.head(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'num_layers': self.num_layers,
                      'num_trees': self.num_trees,
                      'depth': self.depth,
                      'tree_dim': self.tree_dim,
                      'max_features': self.max_features,
                      'input_dropout': self.input_dropout,
                      'choice_function': self.choice_function,
                      'bin_function': self.bin_function,
                      'response_initializer': self.response_initializer,
                      'selection_logits_intializer': self.selection_logits_intializer,
                      'threshold_init_beta': self.threshold_init_beta,
                      'threshold_init_cutoff': self.threshold_init_cutoff,
                      }
        config.update(new_config)
        return config


class NODERegressor(NODE):
    """
    Neural Oblivious Decision Tree (NODE) Regressor model
    based on the NODE architecture proposed by Sergei Popov et al.
    in paper Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs to predict.
        num_layers: `int`, default 8,
            Number of ObliviousDecisionTree layers to use in model
        num_trees: `int`, default 128,
            Number of trees to use in each `ObliviousDecisionTree` layer
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
                 num_outputs: int = 1,
                 num_layers: int = 8,
                 num_trees: int = 16,
                 depth: int = 6,
                 tree_dim: int = 1,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_intializer="random_uniform",
                 threshold_init_beta: float = 1.0,
                 threshold_init_cutoff: float = 1.0,
                 **kwargs):
        super().__init__(num_layers=num_layers,
                         num_trees=num_trees,
                         depth=depth,
                         tree_dim=tree_dim,
                         max_features=max_features,
                         input_dropout=input_dropout,
                         choice_function=choice_function,
                         bin_function=bin_function,
                         response_initializer=response_initializer,
                         selection_logits_intializer=selection_logits_intializer,
                         threshold_init_beta=threshold_init_beta,
                         threshold_init_cutoff=threshold_init_cutoff,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head = layers.Dense(self.num_outputs)

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      }
        config.update(new_config)
        return config


class NODEClassifier(NODE):
    """
    Neural Oblivious Decision Tree (NODE) Classifier model
    based on the NODE architecture proposed by Sergei Popov et al.
    in paper Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        activation_out:
            Activation function to use for the output.
            By default, "sigmoid" is used for binary classification while
            "softmax" is used for multiclass classification.
        num_layers: `int`, default 8,
            Number of ObliviousDecisionTree layers to use in model
        num_trees: `int`, default 128,
            Number of trees to use in each `ObliviousDecisionTree` layer
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
                 num_classes: int = 2,
                 activation_out=None,
                 num_layers: int = 8,
                 num_trees: int = 16,
                 depth: int = 6,
                 tree_dim: int = 1,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_intializer="random_uniform",
                 threshold_init_beta: float = 1.0,
                 threshold_init_cutoff: float = 1.0,
                 **kwargs):
        super().__init__(num_layers=num_layers,
                         num_trees=num_trees,
                         depth=depth,
                         tree_dim=tree_dim,
                         max_features=max_features,
                         input_dropout=input_dropout,
                         choice_function=choice_function,
                         bin_function=bin_function,
                         response_initializer=response_initializer,
                         selection_logits_intializer=selection_logits_intializer,
                         threshold_init_beta=threshold_init_beta,
                         threshold_init_cutoff=threshold_init_cutoff,
                         **kwargs)
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = "sigmoid" if self.num_classes == 1 else "softmax"
        self.head = layers.Dense(self.num_classes,
                                 activation=activation_out)

    def get_config(self):
        config = super().get_config()
        new_config = {'num_classes': self.num_classes,
                      'activation_out': self.activation_out
                      }
        config.update(new_config)
        return config
