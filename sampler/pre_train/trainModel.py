import glob
from mlp import *
 
expDir = '../experiments/1000_1000_500'

print expDir

trainModel(expDir+'/',gpuId='gpu0')
 

