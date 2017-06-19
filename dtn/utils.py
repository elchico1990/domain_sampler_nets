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
		fx, src_fx, src_labels, trg_fx, adda_trg_fx, trg_labels = cPickle.load(f)
		
	src_labels = np.argmax(src_labels,1)
	trg_labels = np.argmax(trg_labels,1)

	print 'Computing T-SNE.'

	model = TSNE(n_components=2, random_state=0)

	print '0'
	TSNE_hA_0 = model.fit_transform(np.vstack((src_fx,fx)))
	#~ print '1'
	#~ TSNE_hA_1 = model.fit_transform(fx)
	#~ print '2'
	#~ TSNE_hA_2 = model.fit_transform(src_fx)
	print '3'
	TSNE_hA_3 = model.fit_transform(np.vstack((src_fx,fx,trg_fx)))
	print '4'
	TSNE_hA_4 = model.fit_transform(np.vstack((src_fx,fx,adda_trg_fx)))
	
	plt.figure(0)
	plt.scatter(TSNE_hA_0[:,0], TSNE_hA_0[:,1], c = np.hstack((src_labels,src_labels)))
	
	plt.figure(1)
	plt.scatter(TSNE_hA_0[:,0], TSNE_hA_0[:,1], c = np.hstack((np.ones((500,)), 2 * np.ones((500,)))))
	
	#~ plt.figure(2)
	#~ plt.scatter(TSNE_hA_1[:,0], TSNE_hA_1[:,1], c = colors_12)
	
	#~ plt.figure(3)
	#~ plt.scatter(TSNE_hA_2[:,0], TSNE_hA_2[:,1], c = colors_12)
	
	plt.figure(4)
	plt.scatter(TSNE_hA_3[:,0], TSNE_hA_3[:,1], c = np.hstack((np.ones((500,)), 2 * np.ones((500,)), 3 * np.ones((500,)))))
	
	plt.figure(5)
	plt.scatter(TSNE_hA_3[:,0], TSNE_hA_3[:,1], c = np.hstack((src_labels,src_labels,trg_labels)))
		
	plt.figure(6)
	plt.scatter(TSNE_hA_4[:,0], TSNE_hA_4[:,1], c = np.hstack((np.ones((500,)), 2 * np.ones((500,)), 3 * np.ones((500,)))))
	
	plt.figure(7)
	plt.scatter(TSNE_hA_4[:,0], TSNE_hA_4[:,1], c = np.hstack((src_labels,src_labels,trg_labels)))
		
	plt.show()

if __name__=='__main__':
	
	computeTSNE()
