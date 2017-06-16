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
		orig_fx, trg_fx, fx, src_labels, trg_labels = cPickle.load(f)

	#~ print np.argmax(src_labels,1).shape
	#~ print trg_labels[:,None].shape
	#~ colors_0 = np.vstack((np.argmax(src_labels,1),trg_labels))
	#~ colors_12 = np.argmax(src_labels,1)
	#~ colors_0 = colors_0[:,None]
	#~ colors_12 = colors_12[:,None]
	colors = np.vstack((np.ones((2000,1)), 2 * np.ones((2000,1))))

	print 'Computing T-SNE.'

	model = TSNE(n_components=2, random_state=0)

	print '0'
	TSNE_hA_0 = model.fit_transform(np.vstack((orig_fx,trg_fx)))
	print '1'
	TSNE_hA_1 = model.fit_transform(np.vstack((orig_fx,fx)))
	#~ print '2'
	#~ TSNE_hA_2 = model.fit_transform(orig_fx)

	plt.figure(0)
	plt.scatter(TSNE_hA_0[:,0], TSNE_hA_0[:,1], c = colors)
	
	
	plt.figure(1)
	plt.scatter(TSNE_hA_1[:,0], TSNE_hA_1[:,1], c = colors)
	

	
	plt.show()

if __name__=='__main__':
	
	computeTSNE()
