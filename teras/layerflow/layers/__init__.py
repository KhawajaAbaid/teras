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


# TabTransformer layers
from .tabtransformer import ColumnEmbedding as TabTransformerColumnEmbedding


# TabNet layers
from .tabnet import (FeatureTransformerBlock as TabNetFeatureTransformerBlock,
                     FeatureTransformer as TabNetFeatureTransformer,
                     Encoder as TabNetEncoder,
                     Decoder as TabNetDecoder)

# SAINT layeres
from teras.layerflow.layers.saint import (SAINTNumericalFeatureEmbedding,
                                          MultiHeadInterSampleAttention,
                                          SAINTTransformer,
                                          Encoder as SAINTEncoder,
                                          ClassificationHead as SAINTClassificationHead,
                                          RegressionHead as SAINTRegressionHead)
