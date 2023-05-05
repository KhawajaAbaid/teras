# Copyright 2023 The Teras Authors
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
from teras.layers.activations import (GLU,
                                      GEGLU)

# NODE layers
from teras.layers.NODE import ObliviousDecisionTree

# TabNet layers
from teras.layers.TabNet import (FeatureTransformerBlock as TabNetFeatureTransformerBlock,
                                 FeatureTransformer as TabNetFeatureTransformer,
                                 Encoder as TabNetEncoder)

# TabTransformer layers
from teras.layers.TabTransformer import (CategoricalFeatureEmbedding as TabTransformerCaterCategoricalFeatureEmbedding,
                                         ColumnEmbedding as TabTransformerColumnEmbedding,
                                         FeedForward as TabTransformerFeedForward,
                                         Transformer as TabTransformerTransformerLayer,
                                         Encoder as TabTransformerEncoder,
                                         RegressionHead as TabTransformerRegressionHead,
                                         ClassificationHead as TabTransformerClassificationHead)

# DNFNet layers
from teras.layers.DNFNet import (DNNF,
                                 FeatureSelection,
                                 Localization)


# SAINT layers
from teras.layers.SAINT import (CategoricalFeaturesEmbedding as SAINTCategoricalFeaturesEmbedding,
                                NumericalFeaturesEmbedding as SAINTNumericalFeaturesEmbedding,
                                FeedForward as SAINTFeedForward,
                                MultiHeadInterSampleAttention as SAINTMultiHeadInterSampleAttention,
                                SAINTTransformer as SAINTTransformer,
                                Encoder as SAINTEncoder,
                                RegressionHead as SAINTRegressionHead,
                                ClassificationHead as SAINTClassificationHead)


# RTDL ResNet layers
from teras.layers.RTDL.ResNet import (ResNetBlock as RTDLResNetBlock,
                                      ClassificationHead as RTDLResNetClassificationHead,
                                      RegressionHead as RTDLResNetRegressionHead)

# RTDL FTTransformer layers
from teras.layers.RTDL.FTTransformer import (FeatureTokenizer as FTFeatureTokenizer,
                                             NumericalFeatureTokenizer as FTNumericalFeatureTokenizer,
                                             CategoricalFeatureTokenizer as FTCategoricalFeatureTokenizer,
                                             Transformer as FTTransformer,
                                             Encoder as FTEncoder,
                                             CLSToken as FTCLSToken,
                                             ClassificationHead as FTClassificationHead,
                                             RegressionHead as FTRegressionHead)

# VIME layers
from teras.layers.VIME import (MaskEstimator as VimeMaskEstimator,
                               FeatureEstimator as VimeFeatureEstimator,
                               Encoder as VimeEncoder,
                               MaskGenerationAndCorruption as VimeMaskGenerationAndCorruption,
                               Predictor as VimePredictor)

# On Embeddings for Numerical Features (paper) layers
from teras.layers.OENF import (PeriodicEmbedding,
                               PiecewiseLinearEncoding)