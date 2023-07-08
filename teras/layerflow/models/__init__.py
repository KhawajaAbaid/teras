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


# Simple Model
from .simple import SimpleModel


# TabTransfomer models
from teras.layerflow.models.tabtransformer import (TabTransformer,
                                                   TabTransformerClassifier,
                                                   TabTransformerRegressor,
                                                   TabTransformerPretrainer)

# TabNet models
from teras.layerflow.models.tabnet import (TabNet,
                                           TabNetRegressor,
                                           TabNetClassifier)


# SAINT models
from teras.layerflow.models.saint import (SAINT,
                                          SAINTClassifier,
                                          SAINTRegressor,
                                          SAINTPretrainer)


# FT Transformer models
from teras.layerflow.models.ft_transformer import (FTTransformer,
                                                   FTTransformerClassifier,
                                                   FTTransformerRegressor)

# DNFNet models
from teras.layerflow.models.dnfnet import (DNFNet,
                                           DNFNetClassifier,
                                           DNFNetRegressor)


# NODE models
from teras.layerflow.models.node import (NODE,
                                         NODEClassifier,
                                         NODERegressor)


# RTDLResNet models
from teras.layerflow.models.rtdl_resnet import (RTDLResNet,
                                                RTDLResNetClassifier,
                                                RTDLResNetRegressor)


# GAIN models
from teras.layerflow.models.gain import (Generator as GAINGenerator,
                                         Discriminator as GAINDiscriminator,
                                         GAIN)

# PCGAIN models
from teras.layerflow.models.gain import (Generator as PCGAINGenerator,
                                         Discriminator as PCGAINDiscriminator)
from teras.layerflow.models.pcgain import (Classifier as PCGAINClassifier,
                                           PCGAIN)


# CTGAN models
from teras.layerflow.models.ctgan import (Generator as CTGANGenerator,
                                          Discriminator as CTGANDiscriminator,
                                          CTGAN)


# TVAE models
from teras.layerflow.models.tvae import (Encoder as TVAEEncoder,
                                         Decoder as TVAEDecoder,
                                         TVAE)
