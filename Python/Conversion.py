"""
Quickly convert the EE JS code to EE Python code for reference
"""
from geemap.conversion import *

input_js = 'd:/Projects/WetlandClassification/JavaScript/YukonPlanetScope.js'
out_js = 'd:/Projects/WetlandClassification/Python/OldClassify.py'

js_to_python(input_js, out_js)
