from itertools import permutations

bands = ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'ndvi', 'swaveHV', 'swaveHVHH', 'ndwi', 'TWIRename', 'ndyi', 'gci',
         'ndei', 'evi2', 'SVI', 'SRI']

comb = [list(permutations(bands))]
combsmall = [l for l in comb if len(l) >= 4]
print(comb)

# This is absurd
# Why did I ever try to do it like this in the first place
# It created a list of 6,402,373,705,728,000 permutations
