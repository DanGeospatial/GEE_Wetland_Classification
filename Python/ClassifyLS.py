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

import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

# Initialize Earth Engine Api
# More information at developers.google.com/earth-engine/guides/python_install-conda#windows
ee.Initialize(project='ee-nelson-remote-sensing')

# ______________________________________________________________________________
# Load EE Assets
# ______________________________________________________________________________

TWI4 = ee.Image("users/danielnelsonca/UndergradThesis/Topographic_Wetness_Index_v4")
TPI2 = ee.Image("users/danielnelsonca/UndergradThesis/Topographic_Position_Index_v2")
newpolygon = ee.FeatureCollection("users/danielnelsonca/UndergradThesis/newpolygon")
Yukon_points_merged = ee.FeatureCollection("users/danielnelsonca/UndergradThesis/Yukon_points_merged")
LS_dataset = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))


# ______________________________________________________________________________
# Filter SAR Data
# ______________________________________________________________________________
# Investigate SAR indexes further

def mask_edge(image):
    edge = image.lt(-30.0)
    masked_image = image.mask().And(edge.Not())
    return image.updateMask(masked_image)


img_vv = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.date('2023-07-01', '2023-07-20'))
    .select('VV')
    .map(mask_edge)
    .mean()
)

img_VH = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.date('2023-07-01', '2023-07-20'))
    .select('VH')
    .map(mask_edge)
    .mean()
)


# ______________________________________________________________________________
# Filter Landsat Images
# ______________________________________________________________________________
def apply_scale_factors(image):
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(
        thermal_bands, None, True
    )


img_LS = (
    LS_dataset
    .filter(ee.Filter.date('2023-07-01', '2023-08-25'))
    .filter(ee.Filter.lessThanOrEquals('CLOUD_COVER', 5))
    .map(apply_scale_factors)
    .median()
)
# ______________________________________________________________________________
# Compute and Rename Indices
# ______________________________________________________________________________
# Normalized Difference Vegetation Index
ndvi = img_LS.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi')
# Normalized Difference Water Index
ndwi = img_LS.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi')
# Normalized Difference Moisture Index
ndmi = img_LS.normalizedDifference(['SR_B5', 'SR_B6']).rename('ndmi')
# Green 1 Chlorophyll Index
GCI = img_LS.expression(
    '(NIR / GREEN) - 1', {
        'NIR': img_LS.select('SR_B5'),
        'GREEN': img_LS.select('SR_B3')
    }).rename('gci')
# Enhanced Vegetation Index
evi2 = img_LS.expression(
    '2.5 * ((NIR - RED) / (NIR + 2.4 * RED + 1))', {
        'NIR': img_LS.select('SR_B5'),
        'RED': img_LS.select('SR_B4')
    }).rename('evi2')
# Standardized Vegetation Index
SVI = img_LS.expression(
    '1.5 * ((NIR - RED) / (NIR - RED + 0.5))', {
        'NIR': img_LS.select('SR_B5'),
        'RED': img_LS.select('SR_B4')
    }).rename('SVI')
# Simple Ratio Index
SRI = img_LS.select('SR_B5').divide(img_LS.select('SR_B4')).rename('SRI')

# add wetness index
TWIRename = TWI4.select('b1').rename('TWIRename')

# Concatenate the ps imagery with the elevation dataset and topographic indexes
img = ee.Image.cat([img_LS, TWIRename, ndvi, ndwi, ndmi, evi2, SVI, SRI, GCI, img_VH, img_vv])
img = img.clip(newpolygon)

# ______________________________________________________________________________
# Run Feature Selection
# ______________________________________________________________________________

# Get all the band names to search
bands = ee.List(
    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'ndvi', 'ndwi', 'TWIRename', 'ndmi', 'gci',
     'evi2', 'SVI', 'SRI', 'VH', 'VV'])

# This property of the table stores the land cover labels.
label = 'ClassID'

# Create training and testing split (80%)
points = Yukon_points_merged.randomColumn()
trainingPoints = points.filter('random <= 0.8')
testPoints = points.filter('random > 0.8')

# Overlay the points on the imagery to get training
training = img.select(bands).sampleRegions(
    collection=trainingPoints,
    properties=[label],
    tileScale=2,
    scale=3
)

# Overlay the points on the imagery to get testing
testing = img.select(bands).sampleRegions(
    collection=testPoints,
    properties=[label],
    tileScale=2,
    scale=3
)

classifier = ee.Classifier.smileRandomForest(
    numberOfTrees=20,
    seed=541
)

testAcc = ee.List([])
varImp = ee.List([])


# Run Recursive Feature Elimination
def subClassifier(inputs):
    global testAcc
    global varImp

    # Train Classifier
    itertrained = classifier.train(
        features=training,
        classProperty=label,
        inputProperties=inputs
    )
    Acc = testing.classify(itertrained).errorMatrix(label, 'classification').accuracy()

    # Get Variable Importance
    dictiter = itertrained.explain()
    iterimportance = ee.Dictionary(dictiter.get('importance'))
    imp = iterimportance.keys()
    values = iterimportance.values()

    # Get the least important variable
    minimumIndex = values.indexOf(values.reduce(ee.Reducer.min()))
    minimumItem = imp.get(minimumIndex)

    inputs = inputs.remove(minimumItem)

    testAcc = testAcc.add(Acc)
    varImp = varImp.add(inputs)

    return inputs


def rfe(subset):
    # Get Test Accuracy
    while subset.length().getInfo() > 3:
        subset = subClassifier(subset)

    global testAcc
    global varImp
    # Find the Bands with the Highest Accuracy
    maxValue = testAcc.indexOf(testAcc.reduce(ee.Reducer.max()))
    maxIndex = varImp.get(maxValue)

    return maxIndex


# Store RFE Output
bestBands = rfe(bands)
print(bestBands.getInfo())

# ______________________________________________________________________________
# Train Random Forest Classifier
# ______________________________________________________________________________

ray.init(include_dashboard=False)


def objective(config):
    ee.Initialize(project='ee-nelson-remote-sensing')
    trained = ee.Classifier.smileRandomForest(
        numberOfTrees=config["numberOfTrees"],
        variablesPerSplit=config["variablesPerSplit"],
        bagFraction=config["bagFraction"],
        seed=541
    ).train(
        features=training,
        classProperty=label,
        inputProperties=bestBands
    )
    testClassify = testing.classify(trained).errorMatrix(label, 'classification').accuracy()
    testnumb = float(testClassify.getInfo())
    return {"acc": testnumb}


initial_params = [{"numberOfTrees": 1000, "variablesPerSplit": 8, "bagFraction": 0.95}]

method = HyperOptSearch(points_to_evaluate=initial_params)
samples = 200
maxvar = int(ee.List(bestBands).length().getInfo())

# Make sure upper bounds of VPS is not higher than input variables
search_config = {
    "numberOfTrees": tune.qrandint(100, 1000, 100),
    "variablesPerSplit": tune.qrandint(1, maxvar, 1),
    "bagFraction": tune.loguniform(0.1, 1)
}

# Reduce the number of concurrent trials to keep GEE happy
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="acc",
        mode="max",
        search_alg=method,
        num_samples=samples,
        max_concurrent_trials=10
    ),
    param_space=search_config,
)
results = tuner.fit()

# These can them be passed to a final classifier
print("Optimal hyperparameters: ", results.get_best_result().config)

# Use optimal hyperparameters and bands to produce metrics
optimaltrained = ee.Classifier.smileRandomForest(
    numberOfTrees=results.get_best_result().config["numberOfTrees"],
    variablesPerSplit=results.get_best_result().config["variablesPerSplit"],
    bagFraction=results.get_best_result().config["bagFraction"],
    seed=541
).train(
    features=training,
    classProperty=label,
    inputProperties=bestBands
)

testClassify = testing.classify(optimaltrained).errorMatrix(label, 'classification')
print(testClassify.getInfo())
print(testClassify.accuracy().getInfo())

# ray stop --force
ray.shutdown()
