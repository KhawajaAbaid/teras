# Copyright 2024 Khawaja Abaid Ullah
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


# Activation
from teras._src.layers.activation.gumbel_softmax import (
    GumbelSoftmax as GumbelSoftmax)

# CTGAN
from teras._src.layers.ctgan.generator_layer import (
    CTGANGeneratorLayer as CTGANGeneratorLayer)
from teras._src.layers.ctgan.discriminator_layer import (
    CTGANDiscriminatorLayer as CTGANDiscriminatorLayer)

# FT-Transformer
from teras._src.layers.ft_transformer.feature_tokenizer import (
    FTTransformerFeatureTokenizer as FTTransformerFeatureTokenizer)

# SAINT
from teras._src.layers.saint.embedding import (
    SAINTEmbedding as SAINTEmbedding)
from teras._src.layers.saint.encoder_layer import (
    SAINTEncoderLayer as SAINTEncoderLayer)
from teras._src.layers.saint.multi_head_inter_sample_attention import (
    SAINTMultiHeadInterSampleAttention as SAINTMultiHeadInterSampleAttention)
from teras._src.layers.saint.projection_head import (
    SAINTProjectionHead as SAINTProjectionHead)
from teras._src.layers.saint.reconstruction_head import (
    SAINTReconstructionHead as SAINTReconstructionHead)

# TabTransformer
from teras._src.layers.tab_transformer.column_embedding import (
    TabTransformerColumnEmbedding as TabTransformerColumnEmbedding
)

# TabNet
from teras._src.layers.tabnet.attentive_transformer import (
    TabNetAttentiveTransformer as TabNetAttentiveTransformer)
from teras._src.layers.tabnet.feature_transformer_layer import (
    TabNetFeatureTransformerLayer as TabNetFeatureTransformerLayer)
from teras._src.layers.tabnet.feature_transformer import (
    TabNetFeatureTransformer as TabNetFeatureTransformer)

# Transformer
from teras._src.layers.transformer.encoder_layer import (
    TransformerEncoderLayer as TransformerEncoderLayer)
from teras._src.layers.transformer.feedforward import (
    TransformerFeedForward as TransformerFeedForward)

# Categorical Embedding
from teras._src.layers.categorical_embedding import (
    CategoricalEmbedding as CategoricalEmbedding)

# Categorical Extraction
from teras._src.layers.categorical_extraction import (
    CategoricalExtraction as CategoricalExtraction)

# CLS Token related
from teras._src.layers.cls_token import CLSToken as CLSToken
from teras._src.layers.cls_token_extraction import (
    CLSTokenExtraction as CLSTokenExtraction)

# Continous Extraction
from teras._src.layers.continuous_extraction import (
    ContinuousExtraction as ContinuousExtraction)

# CutMix
from teras._src.layers.cutmix import CutMix as CutMix

# MixUp
from teras._src.layers.mixup import MixUp as MixUp

# LayerList
from teras._src.layers.layer_list import LayerList as LayerList
