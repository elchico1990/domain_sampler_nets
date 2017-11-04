import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

import cPickle
import utils
from utils import conv_concat, lrelu

class DSN(object):
    """Domain Sampler Network
    """
    def __init__(self, mode='train', learning_rate=0.0003):
        self.mode = mode
        self.learning_rate = learning_rate
	self.hidden_repr_size = 128
    
    def sampler_generator(self, z, y, reuse=False):
	
	'''
	Takes in input noise and labels, and
	generates f_z, which is handled by the 
	net as f(x) was handled. If labels is 
	None, the noise samples are partitioned
	in equal ratios.  
	'''
	
	inputs = tf.concat(axis=1, values=[z, tf.cast(y,tf.float32)])
	#~ inputs = z
	
	with tf.variable_scope('sampler_generator', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.zeros_initializer()):
		
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train_sampler')):
                    
		    net = slim.fully_connected(inputs, 1024, activation_fn = tf.nn.relu, scope='sgen_fc1')
		    net = slim.batch_norm(net, scope='sgen_bn1')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc2')
		    net = slim.batch_norm(net, scope='sgen_bn2')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc3')
		    net = slim.batch_norm(net, scope='sgen_bn3')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn = tf.tanh, scope='sgen_feat')
		    return net
		    
    def E(self, images, reuse=False, make_preds=False, is_training = False, only_output=False, from_features=False):
	
	if only_output == True:
	    with tf.variable_scope('encoder', reuse=reuse):
		with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
			return slim.fully_connected(images, 10, activation_fn=None, scope='fc5')
	
	with tf.variable_scope('encoder', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
		    
		    if (from_features==True):
			net = slim.fully_connected(images, 10, activation_fn=None, scope='fc_output')
			return net
		    
		    
		    net = slim.conv2d(images, 32, 3, scope='conv1')
		    net = slim.conv2d(images, 32, 3, scope='conv11')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
		    net = slim.conv2d(net, 64, 3, scope='conv2')
		    net = slim.conv2d(net, 64, 4, scope='conv22')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
		    net = slim.conv2d(net, 128, 3, scope='conv3')
		    net = slim.conv2d(net, 128, 3, scope='conv33')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool3')
		    net = tf.contrib.layers.flatten(net)
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.tanh, scope='fc4')
		    if (make_preds):
			net = slim.fully_connected(net, 10, activation_fn=None, scope='fc_output')
		    return net
		        		
    def D_e(self, inputs, y, reuse=False):
		
	inputs = tf.concat(axis=1, values=[inputs, tf.cast(y,tf.float32)])
	
	with tf.variable_scope('disc_e',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.zeros_initializer()):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train_sampler')):
                    
		    net = slim.fully_connected(inputs, 128, activation_fn = utils.lrelu, scope='sdisc_fc1')
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='sdisc_prob')
		    
		    return net
	
    def build_model(self):
        
        if self.mode == 'pretrain' or self.mode == 'test' or self.mode == 'train_gen_images':
            
	    self.src_images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'svhn_images')
            self.src_labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
	    self.src_logits = self.E(self.src_images, is_training = True)
		
	    self.src_pred = tf.argmax(self.src_logits, 1)
            self.src_correct_pred = tf.equal(self.src_pred, self.src_labels)
            self.src_accuracy = tf.reduce_mean(tf.cast(self.src_correct_pred, tf.float32))
            
            self.loss = slim.losses.sparse_softmax_cross_entropy(self.src_logits, self.src_labels)
            self.optimizer = tf.train.AdamOptimizer(.001) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
	    
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            src_accuracy_summary = tf.summary.scalar('src_accuracy', self.src_accuracy)
            self.summary_op = tf.summary.merge([loss_summary, src_accuracy_summary])
	
	elif self.mode == 'train_sampler':
				
	    self.images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'svhn_images')
	    self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
	    self.labels = tf.placeholder(tf.int64, [None, 10], 'labels_real')
	    
	    self.fx = self.E(self.images)
	    	
	    self.fzy = self.sampler_generator(self.noise, self.labels) 

	    self.logits_real = self.D_e(self.fx,self.labels, reuse=False) 
	    self.logits_fake = self.D_e(self.fzy,self.labels, reuse=True)
	    
	    self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = self.d_loss_real + self.d_loss_fake
	    
	    self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))
	    
	    self.d_optimizer = tf.train.AdamOptimizer(0.0001)
	    self.g_optimizer = tf.train.AdamOptimizer(0.0001)
	    
	    t_vars = tf.trainable_variables()
	    d_vars = [var for var in t_vars if 'disc_e' in var.name]
	    g_vars = [var for var in t_vars if 'sampler_generator' in var.name]
	    
	    # train op
	    with tf.variable_scope('source_train_op',reuse=False):
		self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=d_vars)
		self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=g_vars)
	    
	    # summary op
	    d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
	    g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
	    self.summary_op = tf.summary.merge([d_loss_summary, g_loss_summary])

	    for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
        
	elif self.mode == 'eval_dsn':
            self.src_noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.src_labels = tf.placeholder(tf.float32, [None, 10], 'labels')
	    self.src_images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'images_trg')
            
            # source domain (svhn to mnist)
            self.fzy = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 

	    self.fx_src = self.E(self.src_images)  
            self.fx_trg = self.E(self.trg_images, reuse=True)
	    
	    self.fzy_logits = self.E(self.fzy,only_output=True)
	    self.fzy_labels = tf.argmax(self.fzy_logits,1)  
	
	elif self.mode == 'end_to_end':

	    self.images = tf.placeholder(tf.float32, [None, 28, 28, 1], 'svhn_images')
            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
	    self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
	    self.balanced_labels = tf.placeholder(tf.int64, [None], 'balanced_labels')
	    
            
	    # Handling images -------------------------------------------------------------------------------
	    
	    self.logits_images = self.E(self.images, is_training = True, make_preds=True)
		
	    self.pred_images = tf.argmax(self.logits_images, 1)
            self.correct_pred_images = tf.equal(self.pred_images, self.labels)
            self.accuracy_images = tf.reduce_mean(tf.cast(self.correct_pred_images, tf.float32))
            
	    self.loss_images = slim.losses.sparse_softmax_cross_entropy(self.logits_images, self.labels)
            self.optimizer_images = tf.train.AdamOptimizer(.0001) 
            self.train_op_images = slim.learning.create_train_op(self.loss_images, self.optimizer_images)
	    
	    # Handling features -------------------------------------------------------------------------------
	    
	    self.fx = self.E(self.images, reuse=True)
	    	
	    self.fzy = self.sampler_generator(self.noise, tf.one_hot(self.labels,10)) 

	    self.logits_features = self.E(self.fzy, reuse=True, is_training = True, from_features=True)
	    
	    # Classification
	    
	    self.loss_features = slim.losses.sparse_softmax_cross_entropy(self.logits_features, self.balanced_labels)
            self.train_op_features = slim.learning.create_train_op(self.loss_features, self.optimizer_images, variables_to_train = [var for var in tf.trainable_variables() if 'fc_output' in var.name])
	    
	    # Adversarial

	    self.logits_real = self.D_e(self.fx, tf.one_hot(self.labels,10), reuse=False) 
	    self.logits_fake = self.D_e(self.fzy, tf.one_hot(self.labels,10), reuse=True)
	    
	    self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = self.d_loss_real + self.d_loss_fake
	    
	    self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))
	    
	    self.d_optimizer = tf.train.AdamOptimizer(0.00001)
	    self.g_optimizer = tf.train.AdamOptimizer(0.0001)
	    
	    # train op
	    with tf.variable_scope('source_train_op',reuse=False):
		self.train_op_d = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=[var for var in tf.trainable_variables() if 'disc_e' in var.name])
		self.train_op_g = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=[var for var in tf.trainable_variables() if 'sampler_generator' in var.name])
	    
	    
	    
	    
	    
	    
            
            # summary op
            loss_images_summary = tf.summary.scalar('loss_images', self.loss_images)
            loss_features_summary = tf.summary.scalar('loss_features', self.loss_features)
            accuracy_images_summary = tf.summary.scalar('accuracy_images', self.accuracy_images)
            self.summary_op = tf.summary.merge([loss_images_summary, accuracy_images_summary])
	        	    
	    # summary op
	    d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
	    g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
	    self.summary_op = tf.summary.merge([d_loss_summary, g_loss_summary])

	    for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
		
		
		
		
		
		
		
		






