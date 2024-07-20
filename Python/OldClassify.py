"""
Here is JavaScript/YukonPlanetScope.js converted directly to Python for reference
This file is not intended to be run!!!
"""

import ee
from ee_plugin import Map

# This is the one to use for PlanetScope (PS_YK)

Map.setCenter(-139.070748, 63.747046)

# Add mosaic collection to map
rgbVis = {'min': 0, 'max': 3000, 'bands': ['b6', 'b4', 'b2']}

Map.addLayer(
PS_YK, rgbVis, 'PS masked',
True)

print(PS_YK, 'PS'.getInfo())

#______________________________________________________________________________
# Compute and Rename Indices
#______________________________________________________________________________
# Normalized Difference Vegetation Index
ndvi = PS_YK.normalizedDifference(['b8', 'b6']).rename('ndvi')
# Normalized Difference Water Index
ndwi = PS_YK.normalizedDifference(['b4', 'b8']).rename ('ndwi')
# Modified Normalized Difference Yellowness Index
NDYI = PS_YK.normalizedDifference(['b5', 'b2']).rename ('ndyi')
# Green 1 Chlorophyll Index
GCI = PS_YK.expression(
'(NIR / GREEN) - 1', {
    'NIR': PS_YK.select('b8'),
    'GREEN': PS_YK.select('b3')
}).rename('gci')
# Yellow Edge Index
NDEI = PS_YK.normalizedDifference(['b7', 'b5']).rename ('ndei')
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

# add DDG
DDGRename = DDG.select('b1').rename('DDGRename')

# add swaveHV
swaveHV = swavesHV.select('b1').rename('swaveHV')

# add swaveHVHH
swaveHVHH = swavesHVHH.select('b1').rename('swaveHVHH')

# Concatenate the ps imagery with the elevation dataset and topographic indexes
img = ee.Image.cat([PS_YK, TWIRename, DDGRename, swaveHV, swaveHVHH, ndvi, ndwi, NDYI, evi2, SVI, SRI, NDEI, GCI])
print(img.getInfo())

img = img.clip(newpolygon)

print(img.getInfo())

#______________________________________________________________________________
# Train Random Forest Classifier
#______________________________________________________________________________

bands = ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'ndvi', 'ndwi', 'TWIRename', 'ndyi', 'gci', 'ndei', 'evi2']
# ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'ndvi', 'swaveHV', 'swaveHVHH', 'ndwi', 'TWIRename', 'ndyi', 'gci', 'ndei', 'evi2', 'SVI', 'SRI']
# ['b4', 'b7', 'b8', 'ndwi', 'ndyi', 'gci', 'ndei', 'TWIRename']
# This property of the table stores the land cover labels.
label = 'ClassID'

# Create training and testing split (80%)
points = Yukon_points.randomColumn()
trainingPoints = points.filter('random <= 0.8')
testPoints = points.filter('random > 0.8')

# Overlay the points on the imagery to get training.
training = img.select(bands).sampleRegions(
collection = trainingPoints,
properties = [label],
tileScale = 2,
scale = 3
)
print(training.getInfo())

# Overlay the points on the imagery to get testing
testing = img.select(bands).sampleRegions(
collection = testPoints,
properties = [label],
tileScale = 2,
scale = 3
)

# Generate Array Subsets:
# This function is bad!
# Do not do this in earth engine
def getCombinations(array):

    def fork(i, t):
        if i == array.length:
            result.push(t)
            return

        fork(i + 1+ t+[array[i]]
        fork(i + 1, t)

    result = []
    fork(0, [])
    return result.slice(0, (result.length - 1))


# Define Classifier and Search List
classifier = ee.Classifier.smileRandomForest(
numberOfTrees = 100,
seed = 541
)
searchList = getCombinations(bands)

# Run Recursive Feature Elimination
def rfe(subset):

    testAcc = ee.List([])
    varImp = ee.List([])

    # Train Classifier
    itertrained = classifier.train(
    features = training,
    classProperty = label,
    inputProperties = subset
    )

    # Get Test Accuracy
    testAcc = testAcc.add(testing.classify(itertrained).errorMatrix(label, 'classification').accuracy())

    # Get Variable Importance
    dictiter = itertrained.explain()
    iterimportance = ee.Dictionary(dictiter.get('importance'))
    varImp = varImp.add(iterimportance.keys())

    return [varImp, testAcc]


# List Helpers
def firstItems(list):
    return ee.List(list[0].flatten())

def secondItems(list):
    return ee.List(list[1].getNumber(0))


# Store RFE Output
rfeControl = searchList.map(rfe)
# Split the Dictionary Back Into Lists
varKeys = rfeControl.map(firstItems)
accKeys = rfeControl.map(secondItems)
# Find the Bands with the Highest Accuracy
varBands = ee.List(varKeys)
varAcc = ee.List(accKeys)
maxValue = varAcc.reduce(ee.Reducer.max())
maxIndex = varAcc.indexOf(maxValue)
# Retreive the Best Band Combination
bestBands = varBands.get(maxIndex)

# Train a SMILE RF classifier with 100 trees on Optimal Variables
trained = ee.Classifier.smileRandomForest(
numberOfTrees = 100,
seed = 541
).train(
features = training,
classProperty = label,
inputProperties = bestBands
)

# Create dictionary that stores the variable importance values from the smileRF classifier
dict = trained.explain()
print('Explain:',dict.getInfo())

variable_importance = ee.Feature(None, ee.Dictionary(dict).get('importance'))
print(variable_importance.getInfo())

# Create Variable Importance Plot
chart = \
ui.Chart.feature.byProperty(variable_importance) \
.setChartType('ColumnChart') \
.setOptions(
title = 'Random Forest Variable Importance',
legend = {position = 'none'},
hAxis = {title = 'Bands'},
vAxis = {title = 'Importance'}
)

print(chart.getInfo())

# Create Confusion Matrix From Test Points
testClassify = testing.classify(trained)
testMatrix = testClassify.errorMatrix(label, 'classification')
print('Test error matrix', testMatrix.getInfo())
print('Test accuracy', testMatrix.accuracy().getInfo())

# Classify the image with the same bands used for training.
classified = img.select(bands).classify(trained)
print(classified.getInfo())

#______________________________________________________________________________
# Add Map and Export
#______________________________________________________________________________

# Prepare For Export
# Define a palette for the IGBP classification.
igbpPalette = [
'aec3d4',
'111149',
'cdb33b',
'cc0013',
'33280d',
'd7cdcc',
'6f6f6f'
]

Map.addLayer(classified,
{'min': 1, 'max': 5, 'palette': igbpPalette},
'classification')


#Export the classification results.
Export.image.toDrive(
image = classified,
description = 'classification',
scale = 15,
region = newpolygon
)
Map