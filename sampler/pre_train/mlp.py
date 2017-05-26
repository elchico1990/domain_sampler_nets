import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import numpy.random as npr
from load import mnist
from utils_optimizers import SGD,RMSprop,Adam
from utils import *

import cPickle

from ConfigParser import *

srng = RandomStreams()


def dropout(X, p=0.):
	retain_prob = p
	X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
	X /= retain_prob
	return X

def model_small(X, w_h, w_h2, w_o, P1, P2):
    X = dropout(X, P1)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, P2)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, P2)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

def model_big(X, w_h, w_h2, w_h3, w_o, P1, P2):
    X = dropout(X, P1)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, P2)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, P2)
    h3 = rectify(T.dot(h2, w_h3))


    h3 = dropout(h3, P2)
    py_x = softmax(T.dot(h3, w_o))
    return h, h2, h3, py_x

def trainModel(expDir,gpuId='gpu0'):
	
	npr.seed(123)


	import theano.sandbox.cuda
	theano.sandbox.cuda.use(gpuId)
	theano.config.floatX = 'float32'

	config = ConfigParser()
	config.read(expDir+'input_configuration')

	mode = config.get('MAIN_PARAMETER_SETTING','mode')
	l_rate = config.getfloat('MAIN_PARAMETER_SETTING','learning_rate')
	prob_input = config.getfloat('MAIN_PARAMETER_SETTING','p_input')
	prob_hidden = config.getfloat('MAIN_PARAMETER_SETTING','p_hidden')
	epochs = config.getint('MAIN_PARAMETER_SETTING','training_epochs')
	
	model_type = config.get('MODEL_PARAMETER_SETTING','model_type')
	h1_size = config.getint('MODEL_PARAMETER_SETTING','h1_size')
	h2_size = config.getint('MODEL_PARAMETER_SETTING','h2_size')
	
	if model_type == 'big_model':
		h3_size = config.getint('MODEL_PARAMETER_SETTING','h3_size')
	
	trX, teX, trY, teY = mnist(onehot=True)

	from theano import tensor as T

	index = T.lvector()
	X = T.fmatrix()
	Y = T.fmatrix()
	P1 = T.scalar()
	P2 = T.scalar()
	LR = T.scalar()

	if model_type == 'small_model':
		
		print 'creating small net'
		
		w_h = init_weights((784, h1_size))
		w_h2 = init_weights((h1_size, h2_size))
		w_o = init_weights((h2_size, 10))

		h, h2, py_x = model_small(X, w_h, w_h2, w_o, P1, P2)
		
		h_eval = theano.function(inputs=[X,P1,P2], outputs = [h2], allow_input_downcast = True)
		
		y_x = T.argmax(py_x, axis=1)

		params = [w_h, w_h2, w_o]
		
	elif model_type == 'big_model':
		
		print 'creating big net'
		
		h3_size = config.getint('MODEL_PARAMETER_SETTING','h3_size')

		w_h = init_weights((784, h1_size))
		w_h2 = init_weights((h1_size, h2_size))
		w_h3 = init_weights((h2_size, h3_size))
		w_o = init_weights((h3_size, 10))

		h, h2, h3, py_x = model_big(X, w_h, w_h2, w_h3, w_o, P1, P2)
		
		h_eval = theano.function(inputs=[X,P1,P2], outputs = [h3], allow_input_downcast = True)
		
		y_x = T.argmax(py_x, axis=1)

		params = [w_h, w_h2, w_h3, w_o]
		
	else:
		raise ValueError('Unknown model type.')
	
	cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
	updates = Adam(cost, params, lr=LR)


	train = theano.function(inputs=[X, Y, P1, P2, LR], outputs=cost, updates=updates, allow_input_downcast=True)
	predict = theano.function(inputs=[X, P1, P2], outputs=y_x, allow_input_downcast=True)

	t = 0

	costs = []
	accTrList = []
	accTeList = []
	
	import time
	
	if prob_input + prob_hidden < 2:
		print mode
	else:
		print 'no_dropout'
	
	for i in range(epochs):
		
		if i == 50:
			print 'updating learning rate'
			l_rate /= 10
		
		ti = time.time()
		for start, end in zip(range(0, 50000, 128), range(128, 50000, 128)):
			cost = train(trX[start:end], trY[start:end], prob_input, prob_hidden, l_rate)
			t = t + 1
			
			if t%100==0:
				costs.append(cost)
		
		accTr = np.mean(np.argmax(trY, axis=1) == predict(trX, 1., 1.))
		accTrList.append(accTr)
		
		accTe = np.mean(np.argmax(teY, axis=1) == predict(teX, 1., 1.))
		accTeList.append(accTe)

		print 'epoch: %d   acc_tr: %.4f   acc_te: %.4f   el_time: %.2f'%(i,accTr,accTe,time.time()-ti)
		print ''%()
		if i%10==0:
			hiddenRepr = h_eval(trX, prob_input, prob_hidden)[0]
			f = file(expDir+'results.pkl', 'w')
			cPickle.dump((costs,accTrList,accTeList,hiddenRepr,trY), f, protocol=cPickle.HIGHEST_PROTOCOL)
			f.close()










