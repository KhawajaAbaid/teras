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

# ===== IMPORTANT NOTE =====
# The concept of this module is copied from Keras-CV, with some
# modifications, as follows:
#   1. In teras, a backbone model is a standalone reusable model that can
#   be used for any compatible downstream task. It can be trained either
#   on its own or as part of another model.
#   2. In teras, we separate out the abstract or partially abstract model
#   classes from backbones and put them in the `blueprints` module
#   instead. And yes this does include the `Backbone` model class. It is
#   done to mainly make it intuitive for the users of the library to
#   understand what to expect from a module, and to avoid unexpected
#   inconsistencies, as one of the main goals of keras is to help
#   accelerate research so making the abstract reusable classes
#   available for use — in an intuitive fashion — should be a must.
# https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/backbones
