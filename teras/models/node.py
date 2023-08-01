from tensorflow import keras
from teras.layers.node import ObliviousDecisionTree
from teras.utils.node import sparsemoid
from teras.activations import sparsemax
from teras.layers.node.node_feature_selector import NodeFeatureSelector
from teras.layerflow.models.node import NODE as _NodeLF
from teras.layers.common.head import ClassificationHead, RegressionHead
from teras.utils.types import UnitsValuesType


@keras.saving.register_keras_serializable("teras.models")
class NODE(_NodeLF):
    """
    Neural Oblivious Decision Tree (NODE) model
    based on the NODE architecture proposed by Sergei Popov et al.
    in paper Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data

    Reference(s):
        https://arxiv.org/abs/1909.06312

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        num_layers: ``int``, default 8,
            Number of ObliviousDecisionTree layers to use in model

        num_trees: ``int``, default 128,
            Number of trees to use in each `ObliviousDecisionTree` layer

        depth: ``int``, default 6,
            Number of splits in every tree

        tree_dim: ``int``, default 1,
            Number of response channels in the response of individual tree

        max_features: ``int``,
            Maximum number of features to use. If None, all features in the input dataset will be used.

        input_dropout: ``float``, default 0.,
            Dropout rate to apply to inputs.

        choice_function:
            Function that computes feature weights s.t. f(tensor, dim).sum(dim) == 1
            By default, ``sparsemax`` is used.

        bin_function:
            Function that computes tree leaf weights.
            By default, ``sparsemoid`` is used.

        response_initializer: default "random_normal",
            Initializer for tree output tensor. Any format that is acceptable by the keras initializers.

        selection_logits_initializer: default "random_uniform",
            Initializer for logits that select features for the tree
            Both thresholds and scales are initialized with data-aware initialization function.

        threshold_init_beta: ``float``, default 1.0,
            Initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        threshold_init_cutoff: ``float``, default 1.0,
            Threshold log-temperatures initializer.
            By default(1.0), log-temperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 8,
                 num_trees: int = 16,
                 depth: int = 6,
                 tree_dim: int = 1,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_initializer="random_uniform",
                 threshold_init_beta: float = 1.0,
                 threshold_init_cutoff: float = 1.0,
                 **kwargs):
        choice_func = sparsemax if choice_function is None else choice_function
        bin_func = sparsemoid if bin_function is None else bin_function

        tree_layers = [ObliviousDecisionTree(num_trees=num_trees,
                                             depth=depth,
                                             tree_dim=tree_dim,
                                             choice_function=choice_func,
                                             bin_function=bin_func,
                                             response_initializer=response_initializer,
                                             selection_logits_initializer=selection_logits_initializer,
                                             threshold_init_beta=threshold_init_beta,
                                             threshold_init_cutoff=threshold_init_cutoff)
                       for _ in range(num_layers)]
        feature_selector = NodeFeatureSelector(data_dim=input_dim,
                                               max_features=max_features,
                                               name="feature_selector")
        dropout = keras.layers.Dropout(input_dropout,
                                       name="input_dropout")
        super().__init__(input_dim=input_dim,
                         tree_layers=tree_layers,
                         feature_selector=feature_selector,
                         dropout=dropout,
                         **kwargs)

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.choice_function = choice_function
        self.response_initializer = response_initializer
        self.selection_logits_initializer = selection_logits_initializer
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'num_layers': self.num_layers,
                  'num_trees': self.num_trees,
                  'depth': self.depth,
                  'tree_dim': self.tree_dim,
                  'max_features': self.max_features,
                  'input_dropout': self.input_dropout,
                  'choice_function': self.choice_function,
                  'bin_function': self.bin_function,
                  'response_initializer': self.response_initializer,
                  'selection_logits_initializer': self.selection_logits_initializer,
                  'threshold_init_beta': self.threshold_init_beta,
                  'threshold_init_cutoff': self.threshold_init_cutoff,
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim, **config)


@keras.saving.register_keras_serializable("teras.models")
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

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Units values to use in the hidden layers in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        num_layers: ``int``, default 8,
            Number of ObliviousDecisionTree layers to use in model

        num_trees: ``int``, default 128,
            Number of trees to use in each `ObliviousDecisionTree` layer

        depth: ``int``, default 6,
            Number of splits in every tree

        tree_dim: ``int``, default 1,
            Number of response channels in the response of individual tree

        max_features: ``int``,
            Maximum number of features to use. If None, all features in the input dataset will be used.

        input_dropout: ``float``, default 0.,
            Dropout rate to apply to inputs.

        choice_function:
            Function that computes feature weights s.t. f(tensor, dim).sum(dim) == 1
            By default, ``sparsemax`` is used.

        bin_function:
            Function that computes tree leaf weights.
            By default, ``sparsemoid`` is used.

        response_initializer: default "random_normal",
            Initializer for tree output tensor. Any format that is acceptable by the keras initializers.

        selection_logits_initializer: default "random_uniform",
            Initializer for logits that select features for the tree
            Both thresholds and scales are initialized with data-aware initialization function.

        threshold_init_beta: ``float``, default 1.0,
            Initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        threshold_init_cutoff: ``float``, default 1.0,
            Threshold log-temperatures initializer.
            By default(1.0), log-temperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
    """
    def __init__(self,
                 num_outputs: int = 1,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 num_layers: int = 8,
                 num_trees: int = 16,
                 depth: int = 6,
                 tree_dim: int = 1,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_initializer="random_uniform",
                 threshold_init_beta: float = 1.0,
                 threshold_init_cutoff: float = 1.0,
                 **kwargs):
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_values,
                              normalization=None)
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         num_trees=num_trees,
                         depth=depth,
                         tree_dim=tree_dim,
                         max_features=max_features,
                         input_dropout=input_dropout,
                         choice_function=choice_function,
                         bin_function=bin_function,
                         response_initializer=response_initializer,
                         selection_logits_initializer=selection_logits_initializer,
                         threshold_init_beta=threshold_init_beta,
                         threshold_init_cutoff=threshold_init_cutoff,
                         head=head,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      'head_units_values': self.head_units_values,
                      }
        config.update(new_config)
        return config


@keras.saving.register_keras_serializable("teras.models")
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

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Units values to use in the hidden layers in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        num_layers: ``int``, default 8,
            Number of ObliviousDecisionTree layers to use in model

        num_trees: ``int``, default 128,
            Number of trees to use in each `ObliviousDecisionTree` layer

        depth: ``int``, default 6,
            Number of splits in every tree

        tree_dim: ``int``, default 1,
            Number of response channels in the response of individual tree

        max_features: ``int``,
            Maximum number of features to use. If None, all features in the input dataset will be used.

        input_dropout: ``float``, default 0.,
            Dropout rate to apply to inputs.

        choice_function:
            Function that computes feature weights s.t. f(tensor, dim).sum(dim) == 1
            By default, ``sparsemax`` is used.

        bin_function:
            Function that computes tree leaf weights.
            By default, ``sparsemoid`` is used.

        response_initializer: default "random_normal",
            Initializer for tree output tensor. Any format that is acceptable by the keras initializers.

        selection_logits_initializer: default "random_uniform",
            Initializer for logits that select features for the tree
            Both thresholds and scales are initialized with data-aware initialization function.

        threshold_init_beta: ``float``, default 1.0,
            Initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        threshold_init_cutoff: ``float``, default 1.0,
            Threshold log-temperatures initializer.
            By default(1.0), log-temperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 num_layers: int = 8,
                 num_trees: int = 16,
                 depth: int = 6,
                 tree_dim: int = 1,
                 max_features: int = None,
                 input_dropout: float = 0.,
                 choice_function=None,
                 bin_function=None,
                 response_initializer="random_normal",
                 selection_logits_initializer="random_uniform",
                 threshold_init_beta: float = 1.0,
                 threshold_init_cutoff: float = 1.0,
                 **kwargs):
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_values,
                                  normalization=None)
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         num_trees=num_trees,
                         depth=depth,
                         tree_dim=tree_dim,
                         max_features=max_features,
                         input_dropout=input_dropout,
                         choice_function=choice_function,
                         bin_function=bin_function,
                         response_initializer=response_initializer,
                         selection_logits_initializer=selection_logits_initializer,
                         threshold_init_beta=threshold_init_beta,
                         threshold_init_cutoff=threshold_init_cutoff,
                         head=head,
                         **kwargs)
        self.num_classes = num_classes
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        new_config = {'num_classes': self.num_classes,
                      'head_units_values': self.head_units_values,
                      }
        config.update(new_config)
        return config
