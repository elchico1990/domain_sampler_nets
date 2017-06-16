import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

import cPickle

from sklearn.manifold import TSNE


def sample_Z(m, n, mode='uniform'):
	if mode=='uniform':
		return npr.uniform(-1., 1., size=[m, n])
	if mode=='gaussian':
		return np.clip(npr.normal(0,0.1,(m,n)),-1,1)

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h
	
def computeTSNE(fileName='./for_tsne.pkl'):
	
	with open(fileName,'r') as f:
		src_fx, fx, src_labels = cPickle.load(f)
		
	colors_0 = np.vstack((np.argmax(src_labels,1),np.argmax(src_labels,1)))
	colors_12 = np.argmax(src_labels,1)
	colors_0 = colors_0[:,None]
	colors_12 = colors_12[:,None]
	colors = np.vstack((np.ones((2000,1)), 2 * np.ones((2000,1))))

	print 'Computing T-SNE.'

	model = TSNE(n_components=2, random_state=0)

	print '0'
	TSNE_hA_0 = model.fit_transform(np.vstack((src_fx,fx)))
	print '2'
	TSNE_hA_2 = model.fit_transform(src_fx)

	plt.figure(0)
	plt.scatter(TSNE_hA_0[:,0], TSNE_hA_0[:,1], c = colors_0)
	
	plt.figure(1)
	plt.scatter(TSNE_hA_0[:,0], TSNE_hA_0[:,1], c = colors)
	
	plt.figure(2)
	plt.scatter(TSNE_hA_2[:,0], TSNE_hA_2[:,1], c = colors_12)
	

	
	plt.show()

if __name__=='__main__':
	
	computeTSNE()
