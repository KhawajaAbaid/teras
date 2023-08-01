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


# General purpose utility functions
from teras.utils.utils import (tf_random_choice,
                               get_normalization_layer,
                               get_categorical_features_cardinalities,
                               get_activation,
                               get_initializer,
                               get_features_metadata_for_embedding,
                               dataframe_to_tf_dataset,
                               convert_dict_to_array_tensor,
                               serialize_layers_collection,
                               deserialize_layers_collection,
                               inject_missing_values,
                               generate_fake_gemstone_data)


# NODE utility function(s)
from teras.utils.node import sparsemoid


# VIME utility function(s)
from teras.utils.vime import (vime_mask_generator,
                              vime_pretext_generator,
                              preprocess_input_vime_self,
                              preprocess_input_vime_semi)
