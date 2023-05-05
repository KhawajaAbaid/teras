from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import tensorflow as tf
from typing import Literal
import math

PERIOD_INITIALIZATIONS = Literal['log-linear', 'normal']


class PeriodicEmbedding(layers.Layer):
    """Period embedding layer for numerical features
    as proposed Yury Gorishniy et al. in the paper
    On Embeddings for Numerical Features in Tabular Deep Learning, 2022.
    Reference(s): https://arxiv.org/abs/2203.05556
    """
    def __init__(self,
                 embedding_dim: int,
                 n_features: int,
                 initialization: PERIOD_INITIALIZATIONS = 'normal',
                 sigma: float = None,
                 **kwargs):
        """
        Args:
            embedding_dim: Dimensionality of numerical embeddings
            n_features: Number of features
            initialization: Initialization strategy.
            sigma: Used for coefficients initialization
        """
        super().__init__(**kwargs)
        assert initialization.lower() in ['normal', 'log-linear'], ("Invalid value for initialization."
                                                                    " Must be one of ['log-linear', 'normal']")
        self.embedding_dim = embedding_dim
        self.n_features = n_features
        self.initialization = initialization.lower()
        self.sigma = sigma

        # The official implementation uses another variable n, that is half of the embedding dim
        self.n = self.embedding_dim // 2

    def build(self, input_shape):
        if self.initialization == 'log-linear':
            self.coefficients = self.sigma ** (tf.range(self.n) / self.n)
            self.coefficients = tf.repeat(self.coefficients[None],
                                          repeats=self.n_features,
                                          axis=1)
        else:
            # initialization must be normal
            self.coefficients = tf.random.normal(shape=(self.n_features, self.n),
                                                 mean=0.,
                                                 stddev=self.sigma)

        self.coefficients = tf.Variable(self.coefficients)

    @staticmethod
    def cos_sin(x):
        return tf.concat([tf.cos(x), tf.sin(x)], -1)

    def call(self, inputs, *args, **kwargs):
        assert inputs.ndim == 2
        pi = tf.constant(math.pi)
        return self.cos_sin(2 * pi * self.coefficients[None] * inputs[..., None])


class PiecewiseLinearEncoding:
    """Implementing this encoding class like that of SKLearn's encoding classes"""
    def __init__(self,
                 n_bins=10,
                 method="quantile",
                 task="classification",
                 tree_params: dict = {},
                 return_type='numpy',
                 encoding='piecewise-linear',
                 ):
        """
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
            feature = X[:, feature_idx]
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
                decision_tree = decision_tree.fit(feature.reshape(-1, 1), y).tree_
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
            feature = X[:, feature_idx]
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
        bins = tf.stack(self.bins, axis=1)
        bins_values = tf.stack(self.bins_values, axis=1)

        return bins_values.numpy()

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)
