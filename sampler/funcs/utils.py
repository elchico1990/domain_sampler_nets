import numpy as np
import theano
import theano.tensor as T
import numpy.random as npr

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def lineTrajectory(x0,x1,num_samples):
	
	x0 = x0[None,:]
	x1 = x1[None,:]
	t = np.linspace(0,1,num_samples)[:,None]
	return t * x1 + (1-t) * x0
	
def circTrajectory(x0,x1,num_samples):
	
	x0 = x0[None,:]
	x1 = x1[None,:]
	t = np.linspace(0,1,num_samples)[:,None]
	return np.cos(np.pi*t/2) * x0 + np.sin(np.pi*t/2) * x1

def lineTrajectory_tensors(x0,x1,num_samples):
	
	x0 = x0[None,:,:,:]
	x1 = x1[None,:,:,:]
	traj = np.zeros((num_samples,x0.shape[1],x0.shape[2],x0.shape[3]))
	
	for t in range(num_samples):
		traj[t] = t * x1 + (1-t) * x0
	
	return traj
	
def circTrajectory_tensors(x0,x1,num_samples):
	
	x0 = x0[None,:,:,:]
	x1 = x1[None,:,:,:]
	traj = np.zeros((num_samples,x0.shape[1],x0.shape[2],x0.shape[3]))
	
	for t in range(num_samples):
		traj[t] = np.cos(np.pi*t/2) * x0 + np.sin(np.pi*t/2) * x1
	
	return traj   
	
def getTrajectoryLength(traj):
	
	return np.sum(np.sqrt(np.sum(np.diff(traj,1,0)**2,1)))
	
def prob(x,gamma,p,mode):
	
	if mode == 'scheduled_dropout':
		return (1.-p)*np.exp(-gamma*x) + p
	
	elif mode == 'annealed_dropout':
		return - (1.-p)*np.exp(-gamma*x) + 1
	
	elif mode == 'regular_dropout':
		return p
	
	elif mode == 'no_dropout':
		return 1.
	
	else:
		raise Exception('Unrecognized mode.')

def computeTSNE(hiddens, labels, expDir, N, fileName):

	print 'Computing 2D t-SNE representation with', str(N), 'random samples.' 

	model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)

	indices = npr.randint(0,len(hiddens)-1,N)

	TSNE_hA = model.fit_transform(hiddens[indices])

	plt.figure()
	plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = labels[indices])
	plt.savefig(expDir+'/'+fileName+'_'+str(N)+'.png')
			
def countActPatterns(hiddens = []):
	for i,h in enumerate(hiddens):
		hiddens[i] = binarize(h)
			
def binarize(X):
	return (X > 0).astype(int)
		
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


if __name__ == '__main__':
	
	x0 = np.array([0,0,0,0])
	x1= np.array([1,1,1,1])
	num_samples = 20
	
	line = lineTrajectory(x0,x1,num_samples)
	circ = circTrajectory(x0,x1,num_samples)
	print getTrajectoryLength(line)
	print getTrajectoryLength(circ)
	print 'break'
