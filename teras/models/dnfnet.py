from tensorflow import keras
from teras.layers.dnfnet.dnnf import DNNF
from teras.utils.types import (IntegerSequence,
                               FloatSequence,
                               UnitsValuesType)
from teras.layers.common.head import (ClassificationHead,
                                      RegressionHead)
from teras.layerflow.models.dnfnet import DNFNet as _DNFNetLF


@keras.saving.register_keras_serializable(package="teras.models")
class DNFNet(_DNFNetLF):
    """
    DNFNet model based on the DNFNet architecture,
    proposed by Liran Katzir et al.
    in the paper,
    "NET-DNF: Effective Deep Modeling Of Tabular Data."

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        num_dnnf_layers: ``int``, default 1,
            Number of ``DNNF`` layers to use in the model

        num_formulas: ``int``, default 256,
            Number of DNF formulas to use in each ``DNNF`` layer.
            Each DNF formula is analogous to a tree in tree based ensembles.

        num_conjunctions_arr: ``List[int]`` or ``Tuple[int]``, default [6, 9, 12, 15],
            Conjunctions array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        conjunctions_depth_arr: ``List[int]`` or ``Tuple[int]``, default [2, 4, 6],
            Conjunctions depth array to use in each ``DNNF`` layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        keep_feature_prob_arr: ``List[float]`` or ``Tuple[float]``, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array to use in each ``DNNF`` layer.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.

        elastic_net_beta: ``float``, default 0.4,
            Used in the computation of Elastic Net Regularization in the ``DNNF`` layer.

        binary_threshold_eps: ``float``, default 1.0,
            Used in the computation of learnable mask in the ``DNNF`` layer.

        temperature: ``float``, default 2.0,
            Temperature value to use in the ``Localization`` layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 input_dim: int,
                 num_dnnf_layers: int = 1,
                 num_formulas: int = 2048,
                 num_conjunctions_arr: IntegerSequence = [6, 9, 12, 15],
                 conjunctions_depth_arr: IntegerSequence = [2, 4, 6],
                 keep_feature_prob_arr: FloatSequence = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 temperature: float = 2.0,
                 **kwargs):
        dnnf_layers = keras.models.Sequential(name="dnnf_layers")
        for _ in range(num_dnnf_layers):
            dnnf_layers.add(DNNF(num_formulas=num_formulas,
                                 num_conjunctions_arr=num_conjunctions_arr,
                                 conjunctions_depth_arr=conjunctions_depth_arr,
                                 keep_feature_prob_arr=keep_feature_prob_arr,
                                 elastic_net_beta=elastic_net_beta,
                                 binary_threshold_eps=binary_threshold_eps,
                                 temperature=temperature))
        super().__init__(input_dim=input_dim,
                         dnnf_layers=dnnf_layers,
                         **kwargs)
        self.input_dim = input_dim
        self.num_dnnf_layers = num_dnnf_layers
        self.num_formulas = num_formulas
        self.num_conjunctions_arr = num_conjunctions_arr
        self.conjunctions_depth_arr = conjunctions_depth_arr
        self.keep_feature_prob_arr = keep_feature_prob_arr
        self.elastic_net_beta = elastic_net_beta
        self.binary_threshold_eps = binary_threshold_eps
        self.temperature = temperature

    def get_config(self):
        config = {'name': self.name,
                  'trainable': self.trainable,
                  'input_dim': self.input_dim,
                  'num_dnnf_layers': self.num_dnnf_layers,
                  'num_formulas': self.num_formulas,
                  'num_conjunctions_arr': self.num_conjunctions_arr,
                  'conjunctions_depth_arr': self.conjunctions_depth_arr,
                  'keep_feature_prob_arr': self.keep_feature_prob_arr,
                  'elastic_net_beta': self.elastic_net_beta,
                  'binary_threshold_eps': self.binary_threshold_eps,
                  'temperature': self.temperature
                  }
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim,
                   **config)


@keras.saving.register_keras_serializable(package="teras.models")
class DNFNetClassifier(DNFNet):
    """
    DNFNetRegressor based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_classes: ``int``, default 2,
            Number of classes to predict.

        head_units_values: ``List[int]`` or ``Tuple[int]``, default None,
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        num_dnnf_layers: ``int``, default 1,
            Number of ``DNNF`` layers to use in the model

        num_formulas: ``int``, default 256,
            Number of DNF formulas to use in each ``DNNF`` layer.
            Each DNF formula is analogous to a tree in tree based ensembles.

        num_conjunctions_arr: ``List[int]`` or ``Tuple[int]``, default [6, 9, 12, 15],
            Conjunctions array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        conjunctions_depth_arr: ``List[int]`` or ``Tuple[int]``, default [2, 4, 6],
            Conjunctions depth array to use in each ``DNNF`` layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        keep_feature_prob_arr: ``List[float]`` or ``Tuple[float]``, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array to use in each ``DNNF`` layer.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.

        elastic_net_beta: ``float``, default 0.4,
            Used in the computation of Elastic Net Regularization in the ``DNNF`` layer.

        binary_threshold_eps: ``float``, default 1.0,
            Used in the computation of learnable mask in the ``DNNF`` layer.

        temperature: ``float``, default 2.0,
            Temperature value to use in the ``Localization`` layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_classes: int = 2,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 num_dnnf_layers: int = 1,
                 num_formulas: int = 2048,
                 num_conjunctions_arr: IntegerSequence = [6, 9, 12, 15],
                 conjunctions_depth_arr: IntegerSequence = [2, 4, 6],
                 keep_feature_prob_arr: FloatSequence = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 temperature: float = 2.0,
                 **kwargs):
        head = ClassificationHead(num_classes=num_classes,
                                  units_values=head_units_values)
        super().__init__(input_dim=input_dim,
                         num_dnnf_layers=num_dnnf_layers,
                         num_conjunctions_arr=num_conjunctions_arr,
                         conjunctions_depth_arr=conjunctions_depth_arr,
                         keep_feature_prob_arr=keep_feature_prob_arr,
                         num_formulas=num_formulas,
                         elastic_net_beta=elastic_net_beta,
                         binary_threshold_eps=binary_threshold_eps,
                         temperature=temperature,
                         head=head,
                         **kwargs)
        self.num_classes = num_classes
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes,
                       'head_units_values': self.head_units_values
                       })
        return config


@keras.saving.register_keras_serializable(package="teras.models")
class DNFNetRegressor(DNFNet):
    """
    DNFNetRegressor based on the DNFNet architecture proposed by Liran Katzir et al.
    in the paper NET-DNF: Effective Deep Modeling Of Tabular Data.

    Reference(s):
        https://openreview.net/forum?id=73WTGs96kho

    Args:
        num_outputs: ``int``, default 1,
            Number of regression outputs.

        head_units_values: ``List[int]`` or ``Tuple[int]``, None,
            Hidden units to use in the Classification head.
            For each value in the list/tuple,
            a hidden layer of that dimensionality is added to the head.
            By default, no hidden layer is used.

        input_dim: ``int``,
            Dimensionality of the input dataset,
            or the number of features in the dataset.

        num_dnnf_layers: ``int``, default 1,
            Number of ``DNNF`` layers to use in the model

        num_formulas: ``int``, default 256,
            Number of DNF formulas to use in each ``DNNF`` layer.
            Each DNF formula is analogous to a tree in tree based ensembles.

        num_conjunctions_arr: ``List[int]`` or ``Tuple[int]``, default [6, 9, 12, 15],
            Conjunctions array to use in each DNNF layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        conjunctions_depth_arr: ``List[int]`` or ``Tuple[int]``, default [2, 4, 6],
            Conjunctions depth array to use in each ``DNNF`` layer.
            It is used in the computation of total number of literals as well as
            computation of number of literals per DNF formula.

        keep_feature_prob_arr: ``List[float]`` or ``Tuple[float]``, default [0.1, 0.3, 0.5, 0.7, 0.9],
            Feature probability array to use in each ``DNNF`` layer.
            It is used by the Feature Selection layer to randomly select a probability
            value that is used in the random selection of input features.

        elastic_net_beta: ``float``, default 0.4,
            Used in the computation of Elastic Net Regularization in the ``DNNF`` layer.

        binary_threshold_eps: ``float``, default 1.0,
            Used in the computation of learnable mask in the ``DNNF`` layer.

        temperature: ``float``, default 2.0,
            Temperature value to use in the ``Localization`` layer.
            According to the paper, The inclusion of an adaptive temperature in this localization mechanism
            facilitates a data-dependent degree of exclusivity:
            at high temperatures, only a few DNNFs will handle an input instance whereas
            at low temperatures, more DNNFs will effectively participate in the ensemble.
    """
    def __init__(self,
                 num_outputs: int = 1,
                 head_units_values: UnitsValuesType = None,
                 input_dim: int = None,
                 num_dnnf_layers: int = 1,
                 num_formulas: int = 2048,
                 num_conjunctions_arr: IntegerSequence = [6, 9, 12, 15],
                 conjunctions_depth_arr: IntegerSequence = [2, 4, 6],
                 keep_feature_prob_arr: FloatSequence = [0.1, 0.3, 0.5, 0.7, 0.9],
                 elastic_net_beta: float = 0.4,
                 binary_threshold_eps: float = 1.0,
                 temperature: float = 2.0,
                 **kwargs):
        head = RegressionHead(num_outputs=num_outputs,
                              units_values=head_units_values)
        super().__init__(input_dim=input_dim,
                         num_dnnf_layers=num_dnnf_layers,
                         num_conjunctions_arr=num_conjunctions_arr,
                         conjunctions_depth_arr=conjunctions_depth_arr,
                         keep_feature_prob_arr=keep_feature_prob_arr,
                         num_formulas=num_formulas,
                         elastic_net_beta=elastic_net_beta,
                         binary_threshold_eps=binary_threshold_eps,
                         temperature=temperature,
                         head=head,
                         **kwargs)
        self.num_outputs = num_outputs
        self.head_units_values = head_units_values

    def get_config(self):
        config = super().get_config()
        new_config = {'num_outputs': self.num_outputs,
                      'head_units_values': self.head_units_values
                      }
        config.update(new_config)
        return config
