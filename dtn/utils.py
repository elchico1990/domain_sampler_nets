import numpy.random as npr
import numpy as np

def sample_Z(m, n):
	return npr.uniform(-1., 1., size=[m, n])

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h
