from tensorflow import keras
from tensorflow.keras import layers
from teras.layers import DNNF


class DNFNetRegressor(keras.models.Model):
    """
    DNFNetRegressor based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_dnnf_layers: Number of DNNF layers to use in the model
        units_out: Number of regression outputs
        n_conjunctions_arr: Conjunctions array. If None, default values will be used.
        conjunctions_depth_arr: Conjunctions depth array. If None, default values will be used.
        keep_feature_prob_arr: Feature probability array. If None. default values will be used.
        n_formulas: Number of DNF formulas. Each DNF formula is analogous to a tree in tree based ensembles.
        elastic_net_beta: Use in the computation of Elastic Net Regularization. Defaults to 0.4
        binary_threshold_eps: Used in the computation of learnable mask. Defaults to 1
    """
    def __init__(self,
                 num_dnnf_layers=1,
                 units_out=1,
                 n_conjunctions_arr=None,
                 conjunctions_depth_arr=None,
                 keep_feature_prob_arr=None,
                 n_formulas=2048,
                 elastic_net_beta=0.4,
                 binary_threshold_eps=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_dnnf_layers = num_dnnf_layers
        self.units_out = units_out
        self.n_conjunctions_arr = n_conjunctions_arr
        self.conjunctions_depth_arr = conjunctions_depth_arr
        self.keep_feature_prob_arr = keep_feature_prob_arr
        self.n_formulas = n_formulas
        self.elastic_net_beta = elastic_net_beta
        self.binary_threshold_eps = binary_threshold_eps

        self.dense_out = layers.Dense(units_out)

    def build(self, input_shape):
        self.dnnf_layers = [
                            DNNF(n_conjunctions_arr=self.n_conjunctions_arr,
                                 conjunctions_depth_arr=self.conjunctions_depth_arr,
                                 keep_feature_prob_arr=self.keep_feature_prob_arr,
                                 n_formulas=self.n_formulas,
                                 elastic_net_beta=self.elastic_net_beta,
                                 binary_threshold_eps=self.binary_threshold_eps)
                            for _ in range(self.num_dnnf_layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.dnnf_layers:
            x = layer(x)
        out = self.dense_out(x)
        return out


class DNFNetClassifier(keras.models.Model):
    """
    DNFNetRegressor based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_dnnf_layers: Number of DNNF layers to use in the model
        num_classes: Number of classes to predict
        n_conjunctions_arr: Conjunctions array. If None, default values will be used.
        conjunctions_depth_arr: Conjunctions depth array. If None, default values will be used.
        keep_feature_prob_arr: Feature probability array. If None. default values will be used.
        n_formulas: Number of DNF formulas. Each DNF formula is analogous to a tree in tree based ensembles.
        elastic_net_beta: Use in the computation of Elastic Net Regularization. Defaults to 0.4
        binary_threshold_eps: Used in the computation of learnable mask. Defaults to 1
        activation_out: Activation function to use for outputs.
            By default, 'sigmoid' will be used for binary and 'softmax' will be used for multiclass classification.
    """
    def __init__(self,
                 num_dnnf_layers=1,
                 num_classes=2,
                 n_conjunctions_arr=None,
                 conjunctions_depth_arr=None,
                 keep_feature_prob_arr=None,
                 n_formulas=2048,
                 elastic_net_beta=0.4,
                 binary_threshold_eps=1,
                 activation_out=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_dnnf_layers = num_dnnf_layers
        self.num_classes = 1 if num_classes <= 2 else num_classes
        self.n_conjunctions_arr = n_conjunctions_arr
        self.conjunctions_depth_arr = conjunctions_depth_arr
        self.keep_feature_prob_arr = keep_feature_prob_arr
        self.n_formulas = n_formulas
        self.elastic_net_beta = elastic_net_beta
        self.binary_threshold_eps = binary_threshold_eps

        self.activation_out = activation_out
        if self.activation_out is None:
            self.activation_out = 'sigmoid' if self.num_classes == 1 else 'softmax'
        self.dense_out = layers.Dense(self.num_classes, activation=self.activation_out)

    def build(self, input_shape):
        self.dnnf_layers = [
                            DNNF(n_conjunctions_arr=self.n_conjunctions_arr,
                                 conjunctions_depth_arr=self.conjunctions_depth_arr,
                                 keep_feature_prob_arr=self.keep_feature_prob_arr,
                                 n_formulas=self.n_formulas,
                                 elastic_net_beta=self.elastic_net_beta,
                                 binary_threshold_eps=self.binary_threshold_eps)
                            for _ in range(self.num_dnnf_layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.dnnf_layers:
            x = layer(x)
        out = self.dense_out(x)
        return out