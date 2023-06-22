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
# limitations under the License.5


# Activation layers
from .activations import (GLU,
                          GEGLU,
                          GumbelSoftmax)


# Embedding layers
from .embedding import CategoricalFeatureEmbedding


# NODE layers
from .node import ObliviousDecisionTree


# TabNet layers
from .tabnet import (AttentiveTransformer as TabNetAttentiveTransformer,
                     FeatureTransformerBlock as TabNetFeatureTransformerBlock,
                     FeatureTransformer as TabNetFeatureTransformer,
                     Encoder as TabNetEncoder,
                     Decoder as TabNetDecoder)


# TabTransformer layers
from .tabtransformer import ColumnEmbedding as TabTransformerColumnEmbedding


# DNFNet layers
from .dnfnet import (DNNF,
                     FeatureSelection,
                     Localization)


# SAINT layers
from .saint import (NumericalFeatureEmbedding as SAINTNumericalFeatureEmbedding,
                    MultiHeadInterSampleAttention as SAINTMultiHeadInterSampleAttention,
                    Encoder as SAINTEncoder)


# RTDL ResNet layers
from teras.layers.rtdl_resnet import (ResNetBlock as RTDLResNetBlock,
                                      ClassificationHead as RTDLResNetClassificationHead,
                                      RegressionHead as RTDLResNetRegressionHead)


# RTDL FTTransformer layers
from teras.layers.ft_transformer import (NumericalFeatureEmbedding as FTNumericalFeatureEmbedding,
                                         CLSToken as FTCLSToken)


# VIME layers
from .vime import (MaskEstimator as VimeMaskEstimator,
                   FeatureEstimator as VimeFeatureEstimator,
                   Encoder as VimeEncoder,
                   MaskGenerationAndCorruption as VimeMaskGenerationAndCorruption,
                   Predictor as VimePredictor)


# On Embeddings for Numerical Features (paper) layers
from .oenf import (PeriodicEmbedding)


# CTGAN layers
from .ctgan import (GeneratorBlock as CTGANGeneratorBlock,
                    DiscriminatorBlock as CTGANDiscriminatorBlock)

