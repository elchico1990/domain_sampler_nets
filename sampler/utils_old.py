import numpy as np
import theano
import theano.tensor as T
import numpy.random as npr

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def computeTSNE(hiddens, labels, expDir, N, fileName):

	print 'Computing 2D t-SNE representation with', str(N), 'random samples.' 

	model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)

	indices = npr.randint(0,len(hiddens)-1,N)

	TSNE_hA = model.fit_transform(hiddens[indices])

	plt.figure()
	plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = labels[indices])
	plt.savefig(expDir+'/'+fileName+'_new_'+str(N)+'.png')
