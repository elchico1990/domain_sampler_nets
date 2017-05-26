import glob
import cPickle
from mlp import *
from utils import *
 
expDir = './experiments/800_200'

with open(expDir+'/results.pkl','r') as f:
	_,_,_,hiddens,labels = cPickle.load(f)
		
computeTSNE(hiddens, np.argmax(labels,1), expDir, N=6000)

 
expDir = './experiments/1000_500'

with open(expDir+'/results.pkl','r') as f:
	_,_,_,hiddens,labels = cPickle.load(f)
		
computeTSNE(hiddens, np.argmax(labels,1), expDir, N=6000)

