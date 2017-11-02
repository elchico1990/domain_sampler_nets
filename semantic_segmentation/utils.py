import numpy as np
import numpy.random as npr
import tensorflow as tf


def sample_Z(m, n, mode='uniform'):
	if mode=='uniform':
		return npr.uniform(-1., 1., size=[m, n])
	if mode=='gaussian':
		return np.clip(npr.normal(0,0.1,(m,n)),-1,1)

def lrelu(input, leak=0.2, scope='lrelu'):
    
    return tf.maximum(input, leak*input)
