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
from teras.legacy.layers.common.head import (ClassificationHead,
                                      RegressionHead)
# -> Common Transformer layers
from teras.legacy.layers.common.transformer import (FeedForward,
                                             Transformer,
                                             Encoder)

# Embedding layers
from teras.legacy.layers.categorical_feature_embedding import \
    CategoricalFeatureEmbedding

# NODE layers
from teras.legacy.layers.node.odst import ObliviousDecisionTree
from teras.legacy.layers.node.node_feature_selector import NodeFeatureSelector

# TabNet layers
from teras.legacy.layers.tabnet.tabnet_attentive_transformer import \
    TabNetAttentiveTransformer
from teras.legacy.layers.tabnet.tabnet_feature_transformer_block import \
    TabNetFeatureTransformerBlock
from teras.legacy.layers.tabnet.tabnet_feature_transformer import \
    TabNetFeatureTransformer
from teras.legacy.layers.tabnet.tabnet_encoder import TabNetEncoder
from teras.legacy.layers.tabnet.tabnet_decoder import TabNetDecoder

# TabTransformer layers
from teras.legacy.layers.tabtransformer.tabtransformer_column_embedding import \
    TabTransformerColumnEmbedding

# DNFNet layers
from teras.legacy.layers.dnfnet.dnfnet_feature_selection import \
    DNFNetFeatureSelection
from teras.legacy.layers.dnfnet.dnfnet_localization import DNFNetLocalization
from teras.legacy.layers.dnfnet.dnnf import DNNF

# SAINT layers
from teras.legacy.layers.saint.saint_numerical_feature_embedding import \
    SAINTNumericalFeatureEmbedding
from teras.legacy.layers.saint.multi_head_inter_sample_attention import \
    MultiHeadInterSampleAttention
from teras.legacy.layers.saint.saint_transformer import SAINTTransformer
from teras.legacy.layers.saint.saint_encoder import SAINTEncoder
from teras.legacy.layers.saint.saint_projection_head import SAINTProjectionHead
from teras.legacy.layers.saint.saint_reconstruction_head import (
    SAINTReconstructionHeadBlock,
    SAINTReconstructionHead)

# RTDL ResNet layers
from teras.legacy.layers.rtdl_resnet.rtdl_resnet_block import RTDLResNetBlock

# RTDL FTTransformer layers
from teras.legacy.layers.ft_transformer.ft_numerical_feature_embedding import \
    FTNumericalFeatureEmbedding
from teras.legacy.layers.ft_transformer.ft_cls_token import FTCLSToken

# VIME layers
from teras.legacy.layers.vime.vime_mask_estimator import VimeMaskEstimator
from teras.legacy.layers.vime.vime_feature_estimator import VimeFeatureEstimator
from teras.legacy.layers.vime.vime_encoder import VimeEncoder
from teras.legacy.layers.vime.vime_predictor import VimePredictor
from teras.legacy.layers.vime.vime_mask_generation_and_corruption import \
    VimeMaskGenerationAndCorruption

# On Embeddings for Numerical Features (paper) layers
from teras.legacy.layers.preprocessing.periodic_embedding import (
    PeriodicEmbedding)

# Regularization layers
from teras.legacy.layers.regularization.mixup import MixUp
from teras.legacy.layers.regularization.cutmix import CutMix

# Encoding layers
from teras.legacy.layers.preprocessing.label_encoding import LabelEncoding

# GAIN layers
from teras.legacy.layers.gain.gain_generator_block import GAINGeneratorBlock
from teras.legacy.layers.gain.gain_discriminator_block import \
    GAINDiscriminatorBlock

# CTGAN layers
from teras.legacy.layers.ctgan.ctgan_generator_block import CTGANGeneratorBlock
from teras.legacy.layers.ctgan.ctgan_discriminator_block import \
    CTGANDiscriminatorBlock

# Normalization layers
from teras.legacy.layers.numerical_feature_normalization import \
    NumericalFeatureNormalization
