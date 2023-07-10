from tensorflow.keras import layers, models
from teras.layers import DNNF, DNFNetClassificationHead, DNFNetRegressionHead
from typing import List

LIST_OF_INT = List[int]
LIST_OF_FLOAT = List[float]


class DNFNet(models.Model):
    """
    DNFNet model based on the DNFNet architecture,
    proposed by Liran Katzir et al.
    in the paper,
    "NET-DNF: Effective Deep Modeling Of Tabular Data."

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_dnnf_layers: `int`, default 1,
            Number of DNNF layers to use in the model
        num_formulas: `int`, default 256,
            Number of DNF formulas to use in each DNNF layer.
            Each DNF formula is analogous to a tree in tree based ensembles.
        num_conjunctions_arr: `List[int]`, default [6, 9, 12, 15],
            Conjunctions array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        conjunctions_depth_arr: `List[int]`, default [2, 4, 6],
            Conjunctions depth array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        keep_feature_prob_arr: `List[float]`, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array to use in each DNNF layer.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.
        elastic_net_beta: `float`, default 0.4,
            Used in the computation of Elastic Net Regularization in the DNNF layer.
        binary_threshold_eps:   `float`, default 1.0,
            Used in the computation of learnable mask in the DNNF layer.
        temperature: `float`, default 2.0,
            Temperature value to use in the Localization layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_dnnf_layers: int = 1,
                 num_formulas: int = 2048,
                 num_conjunctions_arr: LIST_OF_INT = [6, 9, 12, 15],
                 conjunctions_depth_arr: LIST_OF_INT = [2, 4, 6],
                 keep_feature_prob_arr: LIST_OF_FLOAT = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 temperature: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_dnnf_layers = num_dnnf_layers
        self.num_formulas = num_formulas
        self.num_conjunctions_arr = num_conjunctions_arr
        self.conjunctions_depth_arr = conjunctions_depth_arr
        self.keep_feature_prob_arr = keep_feature_prob_arr
        self.elastic_net_beta = elastic_net_beta
        self.binary_threshold_eps = binary_threshold_eps
        self.temperature = temperature
        self.dnnf_layers = models.Sequential(name="dnnf_layers")
        for _ in range(self.num_dnnf_layers):
            self.dnnf_layers.add(DNNF(num_formulas=self.num_formulas,
                                      num_conjunctions_arr=self.num_conjunctions_arr,
                                      conjunctions_depth_arr=self.conjunctions_depth_arr,
                                      keep_feature_prob_arr=self.keep_feature_prob_arr,
                                      elastic_net_beta=self.elastic_net_beta,
                                      binary_threshold_eps=self.binary_threshold_eps))
        self.head = None

    def call(self, inputs):
        outputs = self.dnnf_layers(inputs)
        if self.head is not None:
            outputs = self.head(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        new_config = {'num_dnnf_layers': self.num_dnnf_layers,
                      'num_formulas': self.num_formulas,
                      'num_conjunctions_arr': self.num_conjunctions_arr,
                      'conjunctions_depth_arr': self.conjunctions_depth_arr,
                      'keep_feature_prob_arr': self.keep_feature_prob_arr,
                      'elastic_net_beta': self.elastic_net_beta,
                      'binary_threshold_eps': self.binary_threshold_eps,
                      'temperature': self.temperature}
        config.update(new_config)
        return config


class DNFNetRegressor(DNFNet):
    """
    DNFNetRegressor based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_outputs: `int`, default 1,
            Number of regression outputs.
        num_dnnf_layers: `int`, default 1,
            Number of DNNF layers to use in the model
        num_formulas: `int`, default 256,
            Number of DNF formulas to use in each DNNF layer.
            Each DNF formula is analogous to a tree in tree based ensembles.
        num_conjunctions_arr: `List[int]`, default [6, 9, 12, 15],
            Conjunctions array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        conjunctions_depth_arr: `List[int]`, default [2, 4, 6],
            Conjunctions depth array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        keep_feature_prob_arr: `List[float]`, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array to use in each DNNF layer.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.
        elastic_net_beta: `float`, default 0.4,
            Used in the computation of Elastic Net Regularization in the DNNF layer.
        binary_threshold_eps:   `float`, default 1.0,
            Used in the computation of learnable mask in the DNNF layer.
        temperature: `float`, default 2.0,
            Temperature value to use in the Localization layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 num_dnnf_layers: int = 1,
                 num_formulas: int = 2048,
                 num_conjunctions_arr: LIST_OF_INT = [6, 9, 12, 15],
                 conjunctions_depth_arr: LIST_OF_INT = [2, 4, 6],
                 keep_feature_prob_arr: LIST_OF_FLOAT = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 temperature: float = 2.0,
                 **kwargs):
        super().__init__(num_dnnf_layers=num_dnnf_layers,
                         num_conjunctions_arr=num_conjunctions_arr,
                         conjunctions_depth_arr=conjunctions_depth_arr,
                         keep_feature_prob_arr=keep_feature_prob_arr,
                         num_formulas=num_formulas,
                         elastic_net_beta=elastic_net_beta,
                         binary_threshold_eps=binary_threshold_eps,
                         temperature=temperature,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head = DNFNetRegressionHead(num_outputs=self.num_outputs)

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      }
        config.update(new_config)
        return config


class DNFNetClassifier(DNFNet):
    """
    DNFNetRegressor based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_classes: `int`, default 2,
            Number of classes to predict.
        activation_out:
            Activation function to use for the output layer.
            By default, "sigmoid" is used for binary and "softmax" is used for
            multiclass classification.
        num_dnnf_layers: `int`, default 1,
            Number of DNNF layers to use in the model
        num_formulas: `int`, default 256,
            Number of DNF formulas to use in each DNNF layer.
            Each DNF formula is analogous to a tree in tree based ensembles.
        num_conjunctions_arr: `List[int]`, default [6, 9, 12, 15],
            Conjunctions array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        conjunctions_depth_arr: `List[int]`, default [2, 4, 6],
            Conjunctions depth array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.
        keep_feature_prob_arr: `List[float]`, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array to use in each DNNF layer.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.
        elastic_net_beta: `float`, default 0.4,
            Used in the computation of Elastic Net Regularization in the DNNF layer.
        binary_threshold_eps:   `float`, default 1.0,
            Used in the computation of learnable mask in the DNNF layer.
        temperature: `float`, default 2.0,
            Temperature value to use in the Localization layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_classes: int = 2,
                 activation_out=None,
                 num_dnnf_layers: int = 1,
                 num_formulas: int = 2048,
                 num_conjunctions_arr: LIST_OF_INT = [6, 9, 12, 15],
                 conjunctions_depth_arr: LIST_OF_INT = [2, 4, 6],
                 keep_feature_prob_arr: LIST_OF_FLOAT = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 temperature: float = 2.0,
                 **kwargs):
        super().__init__(num_dnnf_layers=num_dnnf_layers,
                         num_conjunctions_arr=num_conjunctions_arr,
                         conjunctions_depth_arr=conjunctions_depth_arr,
                         keep_feature_prob_arr=keep_feature_prob_arr,
                         num_formulas=num_formulas,
                         elastic_net_beta=elastic_net_beta,
                         binary_threshold_eps=binary_threshold_eps,
                         temperature=temperature,
                         **kwargs)
        self.num_classes = num_classes
        self.activation_out = activation_out
        self.head = DNFNetClassificationHead(num_classes=self.num_classes,
                                             activation_out=self.activation_out)

    def get_config(self):
        config = super().get_config()
        new_config = {'num_classes': self.num_outputs,
                      'activation_out': self.activation_out
                      }
        config.update(new_config)
        return config
