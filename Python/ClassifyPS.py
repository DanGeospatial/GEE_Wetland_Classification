import ee

from itertools import combinations

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

PS_YK = ee.Image("users/danielnelsonca/Projects/YK_composite_PlanetScope")
swavesHV = ee.Image("users/koreenmillard/larch2/RFDI_swave_sHV")
swavesHVHH = ee.Image("users/koreenmillard/larch2/HVHH_swave_sHV")
TWI4 = ee.Image("users/danielnelsonca/UndergradThesis/Topographic_Wetness_Index_v4")
TPI2 = ee.Image("users/danielnelsonca/UndergradThesis/Topographic_Position_Index_v2")
newpolygon = ee.FeatureCollection("users/danielnelsonca/UndergradThesis/newpolygon")
Yukon_points_merged = ee.FeatureCollection("users/danielnelsonca/UndergradThesis/Yukon_points_merged")

# ______________________________________________________________________________
# Compute and Rename Indices
# ______________________________________________________________________________
# Normalized Difference Vegetation Index
ndvi = PS_YK.normalizedDifference(['b8', 'b6']).rename('ndvi')
# Normalized Difference Water Index
ndwi = PS_YK.normalizedDifference(['b4', 'b8']).rename('ndwi')
# Modified Normalized Difference Yellowness Index
NDYI = PS_YK.normalizedDifference(['b5', 'b2']).rename('ndyi')
# Green 1 Chlorophyll Index
GCI = PS_YK.expression(
    '(NIR / GREEN) - 1', {
        'NIR': PS_YK.select('b8'),
        'GREEN': PS_YK.select('b3')
    }).rename('gci')
# Yellow Edge Index
NDEI = PS_YK.normalizedDifference(['b7', 'b5']).rename('ndei')
# Enhanced Vegetation Index
evi2 = PS_YK.expression(
    '2.5 * ((NIR - RED) / (NIR + 2.4 * RED + 1))', {
        'NIR': PS_YK.select('b8'),
        'RED': PS_YK.select('b6')
    }).rename('evi2')
# Standardized Vegetation Index
SVI = PS_YK.expression(
    '1.5 * ((NIR - RED) / (NIR - RED + 0.5))', {
        'NIR': PS_YK.select('b8'),
        'RED': PS_YK.select('b6')
    }).rename('SVI')
# Simple Ratio Index
SRI = PS_YK.select('b8').divide(PS_YK.select('b6')).rename('SRI')

# add wetness index
TWIRename = TWI4.select('b1').rename('TWIRename')

# add swaveHV
swaveHV = swavesHV.select('b1').rename('swaveHV')

# add swaveHVHH
swaveHVHH = swavesHVHH.select('b1').rename('swaveHVHH')

# Concatenate the ps imagery with the elevation dataset and topographic indexes
img = ee.Image.cat([PS_YK, TWIRename, swaveHV, swaveHVHH, ndvi, ndwi, NDYI, evi2, SVI, SRI, NDEI, GCI])
img = img.clip(newpolygon)

# ______________________________________________________________________________
# Run Feature Selection
# ______________________________________________________________________________

# Get all the band names to search
bands = ee.List(
    ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'ndvi', 'swaveHV', 'swaveHVHH', 'ndwi', 'TWIRename', 'ndyi', 'gci',
     'ndei', 'evi2', 'SVI', 'SRI'])

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
    while subset.length().getInfo() > 15:
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
"""
def objective(config):
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
    tune.report({"acc": testClassify})


method = HyperOptSearch()
samples = 1000

# Make sure upper bounds of VPS is not higher than input variables
search_config = {
    "numberOfTrees": tune.qrandint(100, 1000, 100),
    "variablesPerSplit": tune.qrandint(1, 10, 1),
    "bagFraction": tune.loguniform(0.1, 1)
}

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="acc",
        mode="max",
        search_alg=method,
        num_samples=samples
    ),
    param_space=search_config,
)
results = tuner.fit()

print("Optimal hyperparameters: ", results.get_best_result().config)
"""