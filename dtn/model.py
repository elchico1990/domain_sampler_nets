import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

import cPickle


class DSN(object):
    """Domain Sampler Network
    """
    def __init__(self, mode='train', learning_rate=0.0003):
        self.mode = mode
        self.learning_rate = learning_rate
	self.hidden_repr_size = 512
    
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
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn = tf.tanh, scope='sgen_feat')
		    return net
		    
    def E(self, images, reuse=False, make_preds=False, is_training = False):
	
	if images.get_shape()[3] == 3:
	    # For mnist dataset, replicate the gray scale image 3 times.
	    images = tf.image.rgb_to_grayscale(images)
	
	with tf.variable_scope('encoder', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
		    net = slim.conv2d(images, 64, 5, scope='conv1')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
		    net = slim.conv2d(net, 128, 5, scope='conv2')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
		    net = tf.contrib.layers.flatten(net)
		    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.tanh, scope='fc4')
		    if (self.mode == 'pretrain' or self.mode == 'test' or make_preds):
			net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
		    return net
		    
    def ConvDeconv(self, images, reuse=False, is_training=False):
	
	if images.get_shape()[3] == 3:
	    # For mnist dataset, replicate the gray scale image 3 times.
	    images = tf.image.rgb_to_grayscale(images)
	
	with tf.variable_scope('conv_deconv', reuse=reuse):
	    with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], padding='SAME', activation_fn=None,           
				     stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
					activation_fn=tf.nn.relu, is_training=(self.mode=='train_convdeconv')):

		    net = slim.conv2d(images, 64, [3, 3], scope='conv1')   # (batch_size, 16, 16, 64)
		    net = slim.conv2d(net, 128, [3, 3], scope='conv2')     # (batch_size, 8, 8, 128)
		    net = slim.conv2d(net, 256, [3, 3], scope='conv3')     # (batch_size, 4, 4, 256)
		    
		    net = slim.conv2d(net, 64, [4, 4], padding='VALID', scope='conv4', activation_fn=tf.tanh)  # (batch_size, 1, 1, 128)
		    if self.mode != 'train_convdeconv':
			net = tf.contrib.layers.flatten(net)
			return net
		    
		    net = slim.conv2d_transpose(net, 512, [4, 4], padding='VALID', scope='conv_transpose1')   # (batch_size, 4, 4, 512)
		    net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')  # (batch_size, 8, 8, 256)
		    net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')  # (batch_size, 16, 16, 128)
		    
		    net = slim.conv2d_transpose(net, 1, [3, 3], activation_fn=tf.nn.tanh, scope='conv_transpose4')   # (batch_size, 32, 32, 1)
		    
		    return net
		    
    def D_e(self, inputs, y, reuse=False):
	
	#~ x = tf.reshape(x,[-1,128])
	
	inputs = tf.concat(axis=1, values=[inputs, tf.cast(y,tf.float32)])
	
	with tf.variable_scope('disc_e',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.zeros_initializer()):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train_sampler')):
                    
		    #~ net = slim.flatten(inputs)
		    net = slim.fully_connected(inputs, 1024, activation_fn = tf.nn.relu, scope='sdisc_fc1')
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='sdisc_prob')
		    return net

    def build_model(self):
        
        if self.mode == 'train_convdeconv':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
	    self.rec_images = self.ConvDeconv(self.images, is_training = True)
		
	    self.loss = tf.reduce_mean(tf.square(self.rec_images - self.images))
	    
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
	    
            # summary op
            loss_summary = tf.summary.scalar('reconstruction_loss', self.loss)
	    images_summary = tf.summary.image('images', self.images)
	    rec_images_summary = tf.summary.image('reconstructed_images', self.rec_images)
            self.summary_op = tf.summary.merge([loss_summary, images_summary, rec_images_summary])
	        
        if self.mode == 'pretrain' or self.mode == 'test':
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            self.src_labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            self.trg_labels = tf.placeholder(tf.int64, [None], 'mnist_labels')
            
	    self.src_logits = self.E(self.src_images, is_training = True)
		
	    self.src_pred = tf.argmax(self.src_logits, 1)
            self.src_correct_pred = tf.equal(self.src_pred, self.src_labels)
            self.src_accuracy = tf.reduce_mean(tf.cast(self.src_correct_pred, tf.float32))
            
            self.trg_logits = self.E(self.trg_images, is_training = False, reuse=True)
		
	    self.trg_pred = tf.argmax(self.trg_logits, 1)
            self.trg_correct_pred = tf.equal(self.trg_pred, self.trg_labels)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))

            self.loss = slim.losses.sparse_softmax_cross_entropy(self.src_logits, self.src_labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
	    
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            src_accuracy_summary = tf.summary.scalar('src_accuracy', self.src_accuracy)
            trg_accuracy_summary = tf.summary.scalar('trg_accuracy', self.trg_accuracy)
            self.summary_op = tf.summary.merge([loss_summary, src_accuracy_summary, trg_accuracy_summary])
	
	elif self.mode == 'train_sampler':
				
	    self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
	    self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
	    self.labels = tf.placeholder(tf.int64, [None, 10], 'labels_real')
	    
	    self.fx = self.E(self.images)
	    self.fzy = self.sampler_generator(self.noise, self.labels) 

	    self.logits_real = self.D_e(self.fx,self.labels, reuse=False) 
	    self.logits_fake = self.D_e(self.fzy,self.labels, reuse=True)
	    
	    #~ self.d_loss_real = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
	    #~ self.d_loss_fake = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_real))
            
	    self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = self.d_loss_real + self.d_loss_fake
	    
	    #~ self.g_loss = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_real))
            
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
	    self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'images_trg')
            
            # source domain (svhn to mnist)
            self.fzy = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
            self.fx_src = self.E(self.src_images) # instead of extracting the hidden representation from a src image, 
            self.fx_trg = self.E(self.trg_images, reuse=True) # instead of extracting the hidden representation from a src image, 
	    
	    
	    self.h_repr = self.ConvDeconv(self.trg_images)

	elif self.mode == 'train_dsn':
            self.src_noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.src_labels = tf.placeholder(tf.float32, [None, 10], 'labels')
	    self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
	    
	    self.trg_labels = self.E(self.trg_images, make_preds=True)
	    self.trg_labels = tf.one_hot(tf.argmax(self.trg_labels,1),10)
	    
	    self.images = tf.concat(axis=0, values=[self.src_images, tf.image.grayscale_to_rgb(self.trg_images)])
	    self.labels = tf.concat(axis=0, values=[self.src_labels,self.trg_labels])
	    
	    self.orig_src_fx = self.E(self.src_images, reuse=True)
	    
	    self.fzy = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
	    self.fx = self.E(self.images, reuse=True)
	    
	    # E losses
	    
	    self.logits_E_real = self.D_e(self.fzy, self.src_labels)
	    self.logits_E_fake = self.D_e(self.fx, self.labels, reuse=True)
	    
	    
	    
	    #~ self.DE_loss_real = slim.losses.sigmoid_cross_entropy(self.logits_E_real, tf.ones_like(self.logits_E_real))
	    #~ self.DE_loss_fake = slim.losses.sigmoid_cross_entropy(self.logits_E_fake, tf.zeros_like(self.logits_E_fake))
	    self.DE_loss_real = tf.reduce_mean(tf.square(self.logits_E_real - tf.ones_like(self.logits_E_real)))
	    self.DE_loss_fake = tf.reduce_mean(tf.square(self.logits_E_fake - tf.zeros_like(self.logits_E_fake)))
	    
	    self.DE_loss = self.DE_loss_real + self.DE_loss_fake 
	    
	    #~ self.E_loss = slim.losses.sigmoid_cross_entropy(self.logits_E_fake, tf.ones_like(self.logits_E_fake))
	    self.E_loss = tf.reduce_mean(tf.square(self.logits_E_fake - tf.ones_like(self.logits_E_fake)))
	    
	    
	    # Optimizers
	    
            self.DE_optimizer = tf.train.AdamOptimizer(self.learning_rate / 10)
            self.E_optimizer = tf.train.AdamOptimizer(self.learning_rate / 10)
            
            
            t_vars = tf.trainable_variables()
            E_vars = [var for var in t_vars if 'encoder' in var.name]
	    DE_vars = [var for var in t_vars if 'disc_e' in var.name]
            
            
            # train op
            with tf.variable_scope('training_op',reuse=False):
                self.E_train_op = slim.learning.create_train_op(self.E_loss, self.E_optimizer, variables_to_train=E_vars)
                self.DE_train_op = slim.learning.create_train_op(self.DE_loss, self.DE_optimizer, variables_to_train=DE_vars)
            
            # summary op
            E_loss_summary = tf.summary.scalar('E_loss', self.E_loss)
            DE_loss_summary = tf.summary.scalar('DE_loss', self.DE_loss)
            self.summary_op = tf.summary.merge([E_loss_summary, DE_loss_summary])
            

            for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

		    
		
		
		
		

