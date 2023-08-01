# Copyright 2023 Khawaja Abaid Ullah
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Common layers
# -> Common Head layers
from teras.layers.common.head import (ClassificationHead,
                                      RegressionHead)
# -> Common Transformer layers
from teras.layers.common.transformer import (FeedForward,
                                             Transformer,
                                             Encoder)


# Embedding layers
from teras.layers.categorical_feature_embedding import CategoricalFeatureEmbedding


# NODE layers
from teras.layers.node.odst import ObliviousDecisionTree
from teras.layers.node.node_feature_selector import NodeFeatureSelector


# TabNet layers
from teras.layers.tabnet.tabnet_attentive_transformer import TabNetAttentiveTransformer
from teras.layers.tabnet.tabnet_feature_transformer_block import TabNetFeatureTransformerBlock
from teras.layers.tabnet.tabnet_feature_transformer import TabNetFeatureTransformer
from teras.layers.tabnet.tabnet_encoder import TabNetEncoder
from teras.layers.tabnet.tabnet_decoder import TabNetDecoder


# TabTransformer layers
from teras.layers.tabtransformer.tabtransformer_column_embedding import TabTransformerColumnEmbedding


# DNFNet layers
from teras.layers.dnfnet.dnfnet_feature_selection import DNFNetFeatureSelection
from teras.layers.dnfnet.dnfnet_localization import DNFNetLocalization
from teras.layers.dnfnet.dnnf import DNNF


# SAINT layers
from teras.layers.saint.saint_numerical_feature_embedding import SAINTNumericalFeatureEmbedding
from teras.layers.saint.multi_head_inter_sample_attention import MultiHeadInterSampleAttention
from teras.layers.saint.saint_transformer import SAINTTransformer
from teras.layers.saint.saint_encoder import SAINTEncoder
from teras.layers.saint.saint_projection_head import SAINTProjectionHead
from teras.layers.saint.saint_reconstruction_head import (SAINTReconstructionHeadBlock,
                                                          SAINTReconstructionHead)


# RTDL ResNet layers
from teras.layers.rtdl_resnet.rtdl_resnet_block import RTDLResNetBlock


# RTDL FTTransformer layers
from teras.layers.ft_transformer.ft_numerical_feature_embedding import FTNumericalFeatureEmbedding
from teras.layers.ft_transformer.ft_cls_token import FTCLSToken


# VIME layers
from teras.layers.vime.vime_mask_estimator import VimeMaskEstimator
from teras.layers.vime.vime_feature_estimator import VimeFeatureEstimator
from teras.layers.vime.vime_encoder import VimeEncoder
from teras.layers.vime.vime_predictor import VimePredictor
from teras.layers.vime.vime_mask_generation_and_corruption import VimeMaskGenerationAndCorruption


# On Embeddings for Numerical Features (paper) layers
from teras.layers.preprocessing.periodic_embedding import (PeriodicEmbedding)


# Regularization layers
from teras.layers.regularization.mixup import MixUp
from teras.layers.regularization.cutmix import CutMix


# Encoding layers
from teras.layers.preprocessing.label_encoding import LabelEncoding


# GAIN layers
from teras.layers.gain.gain_generator_block import GAINGeneratorBlock
from teras.layers.gain.gain_discriminator_block import GAINDiscriminatorBlock


# CTGAN layers
from teras.layers.ctgan.ctgan_generator_block import CTGANGeneratorBlock
from teras.layers.ctgan.ctgan_discriminator_block import CTGANDiscriminatorBlock


# Normalization layers
from teras.layers.numerical_feature_normalization import NumericalFeatureNormalization
