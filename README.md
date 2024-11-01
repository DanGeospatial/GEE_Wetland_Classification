# GEE_Wetland_Classification
A workflow for classifying wetlands in Google Earth Engine using Landsat 8-9, Sentinel 2 and PlanetScope. 

Important: <br />
You will get bad results if your training points are of poor quality. 
In an ideal workflow one would have training points that represent the full range of spectral values for each class you are trying to distinguish.
Training points should be:
- generated randomly across your AOI (not clustered in specific areas)
- sufficient number of points for each class, especially for those with similar physical properties
- class types should be logical

Features:
- Sentinel 2 Imagery with improved cloud masking using the sentinel2-cloud-detector library
- Computes several indices that can be selected for classification
- Fusion of SAR, Elevation and Multispectral Imagery
- Uses the Random Forest algorithm for Classification
- Incorporates Recursive Feature Elimination

Indices Computed Within Earth Engine (LS & S2):
- NDVI
- NDMI
- MNDWI
- EVI2
- SVI
- SRI

Indices Computed Within Earth Engine (PS):


Indices Computed Within SAGA GIS:
- Topographic Wetness Index
- Topographic Position Index
- Downslope Distance Gradient

Project Description: <br />
This project had two primary goals (1) to refine optimization methods for improved classification accuracy and (2) to compare the capability of three satellite platforms to classify wetlands. In addition to these two primary goals the importance of Short-wave infrared (SWIR) images for wetland classification will also be tested. Landsat 8-9 and Sentinel 2 have SWIR bands whereas and PlanetScope does not. PlanetScope imagery is paid, but it has much higher temporal and spatial resolution than Landsat 8-9 or Sentinel 2. This project was tested in the Yukon, Canada which has experienced more frequent clouding in recent decades. 
Code was switched from using GEE JavaScript to Python. Personally, I think dislike the GEE JavaScript environment. Previous code using JavaScript is available but will not be worked on.

Results: <br />

PlanetScope - 