import glob
import cPickle

import numpy as np

from utils import *

expDir = './experiments/800_200'

with open(expDir+'/results.pkl','r') as f:
	_,_,_,hiddens,labels = cPickle.load(f)

hiddensReal = hiddens[:2000]/hiddens.max()
labelsReal = np.argmax(labels,1)[:2000]



#~ computeTSNE(hiddens, np.argmax(labels,1), expDir, N=6000)

 
expDir = './experiments/800_200'

for i in range(100):
#~ i = 35

	fileName = 'samples_'+str(i)

	with open(expDir+'/'+fileName+'.pkl','r') as f:
		samplesDict = cPickle.load(f)

	sizeSamplesX, sizeSamplesY = samplesDict['0'].shape[0], samplesDict['0'].shape[1]
			
	hiddens = np.zeros((sizeSamplesX * 10, sizeSamplesY))
	labels = np.zeros((sizeSamplesX * 10, 1))

	for i in range(10):
		hiddens[i * sizeSamplesX : i * sizeSamplesX + sizeSamplesX] = samplesDict[str(i)]
		labels[i * sizeSamplesX : i * sizeSamplesX + sizeSamplesX] = i

	hiddens = np.vstack((hiddensReal,hiddens))
	labels = np.vstack((labelsReal[:,None],labels))

	computeTSNE(hiddens, labels, expDir, hiddens.shape[0], fileName)


