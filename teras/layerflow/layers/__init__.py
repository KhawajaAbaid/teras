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


# Atomic layers not specific to a model architecture
from teras.layers.regularization import (MixUp,
                                         CutMix)
from teras.layers.embedding import (CategoricalFeatureEmbedding)

# --> Common transformer layers
from teras.layerflow.layers.common.transformer import (FeedForward,
                                                       Transformer,
                                                       Encoder)



# TabTransformer layers
from teras.layerflow.layers.tabtransformer import (RegressionHead as TabTRegressionHead,
                                                   ClassificationHead as TabTClassificationHead)
from teras.layers.tabtransformer import (ColumnEmbedding as TabTColumnEmbedding)


# TabNet layers
from teras.layerflow.layers.tabnet import (FeatureTransformer as TabNetFeatureTransformer,
                                           Encoder as TabNetEncoder,
                                           Decoder as TabNetDecoder,
                                           ClassificationHead as TabNetClassificationHead,
                                           RegressionHead as TabNetRegressionHead)
from teras.layers.tabnet import (FeatureTransformerBlock as TabNetFeatureTransformerBlock,
                                 AttentiveTransformer as TabNetAttentiveTransformer)


# SAINT layeres
from teras.layerflow.layers.saint import (SAINTTransformer,
                                          Encoder as SAINTEncoder,
                                          ReconstructionHead as SAINTReconstructionHead,
                                          ClassificationHead as SAINTClassificationHead,
                                          RegressionHead as SAINTRegressionHead)
from teras.layers.saint import (NumericalFeatureEmbedding as SAINTNumericalFeatureEmbedding,
                                MultiHeadInterSampleAttention,
                                ReconstructionBlock as SAINTReconstructionBlock)


# FT-Transformer layers
from teras.layerflow.layers.ft_transformer import (NumericalFeatureEmbedding as FTNumericalFeatureEmbedding,
                                                   CLSToken as FTCLSToken,
                                                   ClassificationHead as FTClassificationHead,
                                                   RegressionHead as FTRegressionHead)


# DNFNet layers
from teras.layers.dnfnet import (FeatureSelection as DNFNetFeatureSelection,
                                 Localization as DNFNetLocalization)
from teras.layerflow.layers.dnfnet import (DNNF,
                                           ClassificationHead as DNFNetClassificationHead,
                                           RegressionHead as DNFNetRegressionHead)

# NODE layers
from teras.layers.node import (ObliviousDecisionTree)
from teras.layerflow.layers.node import (ClassificationHead as NODEClassificationHead,
                                         RegressionHead as NODERegressionHead)


# RTDLResNet layers
from teras.layerflow.layers.rtdl_resnet import (ClassificationHead as RTDLResNetClassificationHead,
                                                RegressionHead as RTDLResNetRegressionHead)
from teras.layers.rtdl_resnet import ResNetBlock as RTDLResNetBlock


# GAIN layers
from teras.layers.gain import (GeneratorBlock as GAINGeneratorBlock,
                               DiscriminatorBlock as GAINDiscriminatorBlock)


# PCGAIN layers
# PCGAIN uses same blocks as GAIN
from teras.layers.gain import (GeneratorBlock as PCGAINGeneratorBlock,
                               DiscriminatorBlock as PCGAINGeneratorBlock)


# CTGAN layers
from teras.layers.ctgan import (GeneratorBlock as CTGANGeneratorBlock,
                                DiscriminatorBlock as CTGANDiscriminatorBlock)
