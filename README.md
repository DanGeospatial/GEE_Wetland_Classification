# GEE_Wetland_Classification
A sample workflow for classifying wetlands in Google Earth Engine.

Important:
You will get bad results if your training points are of poor quality. 
In an ideal workflow you will have training points that represent the full range of values for each class you are trying to distinguish.
Training points should be:
-generated randomly across your AOI
-sufficient number of points for each class, especially for those with similar physical properties
-representative of value range for each class; limit selection bias

Features:
- Sentinel 2 Imagery with improved cloud masking using the sentinel2-cloud-detector library
- Computes several indices that can be selected for classification
- Incorporates, SAR, Elevation and Visual Imagery
- Uses the Random Forest algorithm for Classification

Indices Computed Within Earth Engine:
- NDVI
- NDMI
- MNDWI
- EVI2
- SVI
- SRI

Indices Computed Within SAGA GIS:
- Topographic Wetness Index
- Topographic Position Index
- Downslope Distance Gradient

Instructions:

Sources:
