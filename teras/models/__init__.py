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
# limitations under the License.



# NODE models
from .NODE import (NODEClassifier,
                   NODERegressor)

# TabNet models
from .TabNet import (TabNetClassifier,
                     TabNetRegressor)

# TabTransformer models
from .TabTransformer import (TabTransformerClassifier,
                             TabTransformerRegressor)

# DNFNet models
from .DNFNet import (DNFNetRegressor,
                     DNFNetClassifier)

# SAINT models
from .SAINT import (SAINTClassifier,
                    SAINTRegressor)


# RTDL ResNet models
from .RTDL.ResNet import (ResNetClassifier as RTDLResNetClassifier,
                          ResNetRegressor as RTDLResNetRegressor)

# RTDL FTTransformer models
from .RTDL.FTTransformer import (FTTransformerClassifier,
                                 FTTransformerRegressor)

# VIME models
from .VIME import (VimeSelf,
                   VimeSemi)


# CTGAN models
from .CTGAN import (CTGAN,
                    Generator as CTGANGenerator,
                    Discriminator as CTGANDiscriminator,
                    )