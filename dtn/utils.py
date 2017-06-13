import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

import cPickle

from sklearn.manifold import TSNE


def sample_Z(m, n):
	return npr.uniform(-1., 1., size=[m, n])

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h
	
def computeTSNE(fileName='./for_tsne.pkl'):
	
	with open(fileName,'r') as f:
		feat_samples,src_noise,src_labels = cPickle.load(f)

	print 'Computing T-SNE.'

	model = TSNE(n_components=2, random_state=0)

	print feat_samples.shape

	TSNE_hA = model.fit_transform(feat_samples)

	colors = np.argmax(src_labels,1)
	colors = colors[:,None]

	plt.figure()
	plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = colors)
	#~ plt.savefig(self.expDir+'tsne_te_'+str(N)+'.png')
	plt.show()

if __name__=='__main__':
	
	computeTSNE()
