// This script uses code sourced from the Google Earth Engine help documentation.
// This is the one to use for PlanetScope (PS_YK)

Map.setCenter(-139.070748, 63.747046);

// Add mosaic collection to map
var rgbVis = {min: 0, max: 3000, bands: ['b6', 'b4', 'b2']};

Map.addLayer(
    PS_YK, rgbVis, 'PS masked',
    true);

print(PS_YK, 'PS');

//______________________________________________________________________________
// Compute and rename indices
//______________________________________________________________________________
// Normalized Difference Vegetation Index
var ndvi = PS_YK.normalizedDifference(['b8', 'b6']).rename('ndvi');
// Normalized Difference Water Index
var ndwi = PS_YK.normalizedDifference(['b4', 'b8']).rename ('ndwi');
// Modified Normalized Difference Yellowness Index
var NDYI = PS_YK.normalizedDifference(['b5', 'b2']).rename ('ndyi');
// Green 1 Chlorophyll Index 
var GCI = PS_YK.expression(
    '(NIR / GREEN) - 1', {
      'NIR': PS_YK.select('b8'),
      'GREEN': PS_YK.select('b3')
}).rename('gci');
// Yellow Edge Index 
var NDEI = PS_YK.normalizedDifference(['b7', 'b5']).rename ('ndei');
// Enhanced Vegetation Index
var evi2 = PS_YK.expression(
    '2.5 * ((NIR - RED) / (NIR + 2.4 * RED + 1))', {
      'NIR': PS_YK.select('b8'),
      'RED': PS_YK.select('b6')
}).rename('evi2');
// Standardized Vegetation Index
var SVI = PS_YK.expression(
    '1.5 * ((NIR - RED) / (NIR - RED + 0.5))', {
      'NIR': PS_YK.select('b8'),
      'RED': PS_YK.select('b6')
}).rename('SVI');
// Simple Ratio Index
var SRI = PS_YK.select('b8').divide(PS_YK.select('b6')).rename('SRI');

// add wetness index
var TWIRename = TWI4.select('b1').rename('TWIRename');

// add DDG
var DDGRename = DDG.select('b1').rename('DDGRename');

// add swaveHV
var swaveHV = swavesHV.select('b1').rename('swaveHV');

// add swaveHVHH
var swaveHVHH = swavesHVHH.select('b1').rename('swaveHVHH');

// Concatenate the ps imagery with the elevation dataset and topographic indexes
var img = ee.Image.cat([PS_YK, TWIRename, DDGRename, swaveHV, swaveHVHH, ndvi, ndwi, NDYI, evi2, SVI, SRI, NDEI, GCI]);
print(img);

img = img.clip(newpolygon);

print(img);

//______________________________________________________________________________
//______________________________________________________________________________

var bands = ['b4', 'b7', 'b8', 'ndwi', 'ndyi', 'gci', 'ndei', 'TWIRename'];
// ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'ndvi', 'swaveHV', 'swaveHVHH', 'ndwi', 'TWIRename', 'ndyi', 'gci', 'ndei', 'evi2', 'SVI', 'SRI']
// This property of the table stores the land cover labels.
var label = 'ClassID';

// Create training and testing split (80%)
var points = Yukon_points.randomColumn();
var trainingPoints = points.filter('random <= 0.9');
var testPoints = points.filter('random > 0.9');

// Overlay the points on the imagery to get training.
var training = img.select(bands).sampleRegions({
  collection: trainingPoints,
  properties: [label],
  tileScale: 2,
  scale: 3
});
print(training);

// Overlay the points on the imagery to get testing
var testing = img.select(bands).sampleRegions({
  collection: testPoints,
  properties: [label],
  tileScale: 2,
  scale: 3
});

// Train a SMILE RF classifier with 1000 trees
var trained = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  seed: 541
}).train({
  features: training, 
  classProperty: label, 
  inputProperties: bands
});

// Create dictionary that stores the variable importance values from the smileRF classifier
var dict = trained.explain();
print('Explain:',dict);
 
var variable_importance = ee.Feature(null, ee.Dictionary(dict).get('importance'));
print(variable_importance);

// Create Variable Importance Plot
var chart =
  ui.Chart.feature.byProperty(variable_importance)
    .setChartType('ColumnChart')
    .setOptions({
      title: 'Random Forest Variable Importance',
      legend: {position: 'none'},
      hAxis: {title: 'Bands'},
      vAxis: {title: 'Importance'}
});

print(chart); 

// Create Confusion Matrix From Test Points
var testClassify = testing.classify(trained);
var testMatrix = testClassify.errorMatrix(label, 'classification');
print('Test error matrix', testMatrix);
print('Test accuracy', testMatrix.accuracy());

// Classify the image with the same bands used for training.
var classified = img.select(bands).classify(trained);
print(classified);

//______________________________________________________________________________
//______________________________________________________________________________

// Prepare For Export
// TODO these colors aren't good
// Define a palette for the IGBP classification.
var igbpPalette = [
  'aec3d4', 
  '111149', 
  'cdb33b', 
  'cc0013', 
  '33280d', 
  'd7cdcc', 
  '6f6f6f'  
  ];

Map.addLayer(classified,
             {min: 1, max: 5, palette: igbpPalette},
             'classification');
             

//Export the classification results.
Export.image.toDrive({
  image: classified,
  description: 'classification',
  scale: 15,
  region: newpolygon
});
