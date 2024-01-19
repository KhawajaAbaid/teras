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


# NODE models
from teras.legacy.models.node import (NODE,
                               NODEClassifier,
                               NODERegressor)

# TabNet models
from teras.legacy.models.tabnet import (TabNet,
                                 TabNetClassifier,
                                 TabNetRegressor,
                                 TabNetPretrainer)

# TabTransformer models
from teras.legacy.models.tabtransformer import (TabTransformer,
                                         TabTransformerClassifier,
                                         TabTransformerRegressor)

# DNFNet models
from teras.legacy.models.dnfnet import (DNFNet,
                                 DNFNetRegressor,
                                 DNFNetClassifier)

# SAINT models
from teras.legacy.models.saint import (SAINT,
                                SAINTClassifier,
                                SAINTRegressor)

# RTDL ResNet models
from teras.legacy.models.rtdl_resnet import (RTDLResNetClassifier,
                                      RTDLResNetRegressor)

# RTDL FTTransformer models
from teras.legacy.models.ft_transformer import (FTTransformer,
                                         FTTransformerClassifier,
                                         FTTransformerRegressor)

# VIME models
from teras.legacy.models.vime import (VimeSelf,
                               VimeSemi)

# CTGAN models
from teras.legacy.models.ctgan import (CTGANGenerator,
                                CTGANDiscriminator,
                                CTGAN)

# TVAE models
from teras.legacy.models.tvae import (TVAEEncoder,
                               TVAEDecoder,
                               TVAE)

# GAIN models
from teras.legacy.models.gain import (GAINGenerator,
                               GAINDiscriminator,
                               GAIN)

# PCGAIN models
from teras.legacy.models.pcgain import (PCGAINGenerator,
                                 PCGAINDiscriminator,
                                 PCGAINClassifier,
                                 PCGAIN)
