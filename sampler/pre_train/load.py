import numpy as np
import os


datasets_dir = '/home/mzanotto/Data/computer_vision/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY

def cifar10(ntrain=60000,ntest=10000,onehot=True):
	
	trX = np.zeros((50000,3072))
	trY = np.zeros((50000,10))
	
	for i in range(1,6):
		f = unpickle(datasets_dir+'/cifar10/data_batch_'+str(i))
		trX[(i-1)*10000:i*10000,:] = f['data']
		trY[(i-1)*10000:i*10000,:] = one_hot(f['labels'],10)

	f = unpickle(datasets_dir+'cifar10/test_batch')
	
	teX = f['data']
	teY = one_hot(f['labels'],10)
	
	import random
	indices = range(len(trX))
	random.shuffle(indices)
	trX = trX[indices]
	trY = trY[indices]
	
	trX = trX/255.
	teX = teX/255.

	return trX,teX,trY,teY
	
if __name__== '__main__':

	cifar10()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
