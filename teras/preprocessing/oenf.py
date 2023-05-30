import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np


class PiecewiseLinearEncoding:
    """
    PiecewiseLinearEncoding class for encoding numerical features
    as proposed Yury Gorishniy et al. in the paper
    On Embeddings for Numerical Features in Tabular Deep Learning, 2022.

    We implement it like those encoding classes in sklearn.
    It expects the input dataset to be a pandas dataframe.

    Reference(s):
        https://arxiv.org/abs/2203.05556

    Args:
        X: input dataset
        y: target feature or feature name.
            must be specified in case of supervised (tree-based) encoding
        method: quantile, tree
        tree_params: a dictionary of parameters for sklearn's decision tree
        return_type: numpy or tensor.
            numpy returns a numpy array while tensor returns tensorflow tensor
        encoding: piecewise-linear, binary
    """
    def __init__(self,
                 n_bins=10,
                 method="quantile",
                 task="classification",
                 tree_params: dict = {},
                 encoding='piecewise-linear',
                 ):
        self.n_bins = n_bins
        self.method = method
        self.task = task
        self.tree_params = tree_params
        self.encoding = encoding
        self.bin_edges = []

    def fit(self,
            X,
            y=None):
        for feature_idx in range(X.shape[1]):
            feature = X.iloc[:, feature_idx]
            # Discretizing using quantiles
            if self.method.lower() == "quantile":
                feature_n_bins = min(self.n_bins, tf.unique(feature)[0].shape[0])
                quantiles = tf.unique(tfp.stats.quantiles(feature, feature_n_bins))[0]
                self.bin_edges.append(quantiles.numpy())

            elif self.method.lower() == "tree":
                assert y is not None, ("For tree based encoding, y must not be None. Given y: ", y)
                if isinstance(y, str):
                    # y is a feature name that must be present in the feature matrix X
                    # otherwise it's assumed to be a feature itself
                    y = X[y]
                if self.task == "regression":
                    decision_tree = DecisionTreeRegressor(max_leaf_nodes=self.n_bins,
                                                          **self.tree_params)
                else:
                    decision_tree = DecisionTreeClassifier(max_leaf_nodes=self.n_bins,
                                                           **self.tree_params)
                # decision_tree = decision_tree.fit(feature.reshape(-1, 1), y).tree_
                decision_tree = decision_tree.fit(feature.values.reshape(-1, 1), y).tree_
                tree_thresholds = []
                for node_id in range(decision_tree.node_count):
                    # the following condition is True only for split nodes
                    # See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
                    if decision_tree.children_left[node_id] != decision_tree.children_right[node_id]:
                        tree_thresholds.append(decision_tree.threshold[node_id])
                    tree_thresholds.append(feature.min())
                    tree_thresholds.append(feature.max())
                    self.bin_edges.append(np.array(sorted(set(tree_thresholds))))

    def transform(self,
                  X):
        """Transforms a given dataset based on the encodings learned in fit"""
        self.bins = []
        self.bins_values = []
        for feature_idx in range(X.shape[1]):
            feature = X.iloc[:, feature_idx]
            feature_bin_edges = np.r_[-np.inf, self.bin_edges[feature_idx][1:-1], np.inf]
            self.bins.append(
                tf.cast(
                    tf.searchsorted(
                        feature_bin_edges,
                        feature,
                        side="right"
                        ),
                    dtype="int32").numpy() - 1
            )

            if self.encoding == 'binary':
                self.bins_values.append(
                    tf.ones_like(
                        feature
                    )
                )
            else:
                # encoding must be piecewise-linear
                feature_bin_sizes = self.bin_edges[feature_idx][1:] - self.bin_edges[feature_idx][:-1]
                feature_bins = self.bins[feature_idx]

                self.bins_values.append(
                    (feature - self.bin_edges[feature_idx][feature_bins]) / feature_bin_sizes[feature_bins]
                )

        # n_bins = len(self.bin_edges) - 1
        # bins = tf.stack(self.bins, axis=1)
        bins_values = tf.stack(self.bins_values, axis=1)
        return bins_values.numpy()

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)