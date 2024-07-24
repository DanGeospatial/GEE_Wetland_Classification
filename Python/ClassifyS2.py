"""
 Copyright 2024 Daniel Nelson

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import ee
import eemont

import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

# Initialize Earth Engine Api
# More information at developers.google.com/earth-engine/guides/python_install-conda#windows
ee.Initialize()

# ______________________________________________________________________________
# Load EE Assets
# ______________________________________________________________________________

newpolygon = ee.FeatureCollection("users/danielnelsonca/UndergradThesis/newpolygon")
Yukon_points_merged = ee.FeatureCollection("users/danielnelsonca/UndergradThesis/Yukon_points_merged")
TWI4 = ee.Image("users/danielnelsonca/UndergradThesis/Topographic_Wetness_Index_v4")
TPI2 = ee.Image("users/danielnelsonca/UndergradThesis/Topographic_Position_Index_v2")

