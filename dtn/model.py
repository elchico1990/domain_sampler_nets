import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

import cPickle

from utils import knn

class DSN(object):
    """Domain Sampler Net
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
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn = tf.tanh, scope='sgen_feat')
		    return net
		    
    def E(self, images, reuse=False, make_preds=False, is_training = False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        if images.get_shape()[3] == 1:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)
        
	with tf.variable_scope('encoder', reuse=reuse):
	    net = slim.dropout(images, 0.9, is_training=(self.mode=='pretrain' and is_training == True),scope='dropout1')
	    net = slim.conv2d(net, 64, [5, 5], scope='conv1')
	    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
	    net = slim.conv2d(net, 128, [5, 5], scope='conv2')
	    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
	    net = slim.flatten(net)
	    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.tanh, scope='fc3')
	    
	    if (self.mode == 'pretrain' or self.mode == 'test' or make_preds):
		net = slim.dropout(net, 0.5, is_training=(self.mode=='pretrain' and is_training == True),scope='dropout3')
		net = slim.fully_connected(net, 10, activation_fn=tf.sigmoid,scope='fc4')
		
	    return net
	
    def C(self, features, reuse=False, is_training = False):
        
        with tf.variable_scope('classifier', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train_classifier' and is_training == True)):
                    
                    net = slim.fully_connected(features, 1024, scope='fc1') 
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.fully_connected(net, 1024, scope='fc2')
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.fully_connected(net, 10, activation_fn=tf.sigmoid, scope='out')
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
	    
    def G(self, inputs, reuse=False):
        # inputs: (batch, 1, 1, 128)
	
	if inputs.get_shape()[1] != 1:
	    inputs = tf.expand_dims(inputs, 1)
	    inputs = tf.expand_dims(inputs, 1)
	
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=tf.nn.tanh,           
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.tanh, is_training=(self.mode=='train_dsn')):

                    net = slim.conv2d_transpose(inputs, 512, [4, 4], padding='VALID', scope='conv_transpose1')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')  # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d_transpose(net, 1, [3, 3], scope='conv_transpose4')   # (batch_size, 32, 32, 1)
                    return net
    
    def D_g(self, images, reuse=False):
	
	if images.get_shape()[3] == 3:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.rgb_to_grayscale(images)
	
        # images: (batch, 32, 32, 1)
        with tf.variable_scope('disc_g', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    
                    net = slim.conv2d(images, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv1')   # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv2')   # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 512, [3, 3], scope='conv3')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.flatten(net)
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='fc1')   # (batch_size, 3)
                    return net
	    
    def build_model(self):
        
        if self.mode == 'pretrain' or self.mode == 'test':
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'usps_images')
            self.src_labels = tf.placeholder(tf.int64, [None], 'mnist_labels')
            self.trg_labels = tf.placeholder(tf.int64, [None], 'usps_labels')
            
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
				
	    self.images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
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
	    self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'images_trg')
            
            # source domain (svhn to mnist)
            self.fzy = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
            self.fx_src = self.E(self.src_images) # instead of extracting the hidden representation from a src image, 
            self.fx_trg = self.E(self.trg_images, reuse=True) # instead of extracting the hidden representation from a src image, 

	elif self.mode == 'train_dsn':
            self.src_noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            
	    self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'src_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'trg_images')
	    
	    self.src_labels = tf.placeholder(tf.float32, [None, 10], 'src_labels')
	    self.trg_labels = tf.placeholder(tf.float32, [None, 10], 'trg_labels')
	    
	    self.trg_labels_inf = self.E(self.trg_images, make_preds=True)
	    self.trg_labels_oh = tf.one_hot(tf.argmax(self.trg_labels_inf,1),10)
	    
	    #~ self.images = tf.concat(axis=0, values=[tf.image.grayscale_to_rgb(self.src_images), tf.image.grayscale_to_rgb(self.trg_images)])
	    self.images = tf.concat(axis=0, values=[self.src_images,self.trg_images])
	    self.labels = tf.concat(axis=0, values=[self.src_labels,self.trg_labels])
	    
	    self.orig_src_fx = self.E(self.src_images, reuse=True)
	    
	    self.fzy = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
	    self.fx = self.E(self.images, reuse=True)
	    
	    self.trg_pred_knn = knn(self.fx, self.fzy, self.src_labels, K = 5)
	    
	    
	    self.GE_trg = self.G(self.E(self.trg_images, reuse=True))
	    #~ self.GE_trg = self.G(self.E(self.GE_trg, reuse=True), reuse=True)
	    
	    #~ self.EG_fzy = self.E(self.G(self.fzy, reuse=True), reuse=True)
	    
	    
	    self.gen_trg_images = self.G(self.fzy, reuse=True)
	    
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
	    
	    # G losses
	    
	    self.logits_G_real = self.D_g(self.trg_images)
	    self.logits_G_fake = self.D_g(self.gen_trg_images, reuse=True)
	    
	    #~ self.DG_loss_real = slim.losses.sigmoid_cross_entropy(self.logits_G_real, tf.ones_like(self.logits_G_real))
	    #~ self.DG_loss_fake = slim.losses.sigmoid_cross_entropy(self.logits_G_fake, tf.zeros_like(self.logits_G_fake))
	    self.DG_loss_real = tf.reduce_mean(tf.square(self.logits_G_real - tf.ones_like(self.logits_G_real)))
	    self.DG_loss_fake = tf.reduce_mean(tf.square(self.logits_G_fake - tf.zeros_like(self.logits_G_fake)))
	    
	    self.DG_loss = self.DG_loss_real + self.DG_loss_fake 
	    
	    #~ self.G_loss = slim.losses.sigmoid_cross_entropy(self.logits_G_fake, tf.ones_like(self.logits_G_fake))
	    self.G_loss = tf.reduce_mean(tf.square(self.logits_G_fake - tf.ones_like(self.logits_G_fake)))
	    
	    # Trg const loss
	    
	    self.const_loss = tf.reduce_mean(tf.square(self.GE_trg - tf.image.grayscale_to_rgb(self.trg_images))) * .15 #+ tf.reduce_mean(tf.square(self.EG_fzy - self.fzy)) * 15
	    
	    
	    # Optimizers
	    
            self.DE_optimizer = tf.train.AdamOptimizer(self.learning_rate / 10)
            self.E_optimizer = tf.train.AdamOptimizer(self.learning_rate / 10)
            self.DG_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.G_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.const_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            
            t_vars = tf.trainable_variables()
            E_vars = [var for var in t_vars if 'encoder' in var.name]
            DE_vars = [var for var in t_vars if 'disc_e' in var.name]
            G_vars = [var for var in t_vars if 'generator' in var.name]
            DG_vars = [var for var in t_vars if 'disc_g' in var.name]
            
            # train op
            with tf.variable_scope('training_op',reuse=False):
                self.E_train_op = slim.learning.create_train_op(self.E_loss, self.E_optimizer, variables_to_train=E_vars)
                self.DE_train_op = slim.learning.create_train_op(self.DE_loss, self.DE_optimizer, variables_to_train=DE_vars)
                self.G_train_op = slim.learning.create_train_op(self.G_loss, self.G_optimizer, variables_to_train=G_vars)
                self.DG_train_op = slim.learning.create_train_op(self.DG_loss, self.DG_optimizer, variables_to_train=DG_vars)
                self.const_train_op = slim.learning.create_train_op(self.const_loss, self.const_optimizer, variables_to_train=E_vars+G_vars)
            
            # summary op
            E_loss_summary = tf.summary.scalar('E_loss', self.E_loss)
            DE_loss_summary = tf.summary.scalar('DE_loss', self.DE_loss)
            G_loss_summary = tf.summary.scalar('G_loss', self.G_loss)
            DG_loss_summary = tf.summary.scalar('DG_loss', self.DG_loss)
            gen_trg_images_summary = tf.summary.image('gen_trg_images', self.gen_trg_images, max_outputs=24)
            self.summary_op = tf.summary.merge([E_loss_summary, DE_loss_summary, 
                                                    G_loss_summary, DG_loss_summary,
						    gen_trg_images_summary])
            

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)  
	
	elif self.mode == 'train_classifier':
	    
	    self.src_noise = tf.placeholder(tf.float32, [None, 100], 'noise')
	    self.src_labels = tf.placeholder(tf.int64, [None, 10], 'src_labels')
	    self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'src_images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'trg_images')
	    self.trg_labels = tf.placeholder(tf.int64, [None, 10], 'trg_labels')

	    self.fzy = self.sampler_generator(self.src_noise, self.src_labels)
	    self.fx_src = self.E(self.src_images) 
	    self.fx_trg = self.E(self.trg_images, reuse=True)

	    self.fzy_logits = self.C(self.fzy, is_training = True)
	    self.fzy_pred = tf.argmax(self.fzy_logits, 1)
	    self.fzy_correct_pred = tf.equal(self.fzy_pred, tf.argmax(self.src_labels,1))
	    self.fzy_accuracy = tf.reduce_mean(tf.cast(self.fzy_correct_pred, tf.float32))

	    self.fx_src_logits = self.C(self.fx_src, is_training = False, reuse=True)
	    self.fx_src_pred = tf.argmax(self.fx_src_logits, 1)
	    self.fx_src_correct_pred = tf.equal(self.fx_src_pred, tf.argmax(self.src_labels,1))
	    self.fx_src_accuracy = tf.reduce_mean(tf.cast(self.fx_src_correct_pred, tf.float32))

	    self.fx_trg_logits = self.C(self.fx_trg, is_training = False, reuse=True)
	    self.fx_trg_pred = tf.argmax(self.fx_trg_logits, 1)
	    self.fx_trg_correct_pred = tf.equal(self.fx_trg_pred, tf.argmax(self.trg_labels,1))
	    self.fx_trg_accuracy = tf.reduce_mean(tf.cast(self.fx_trg_correct_pred, tf.float32))

	    self.loss = slim.losses.sparse_softmax_cross_entropy(self.fzy_logits, tf.argmax(self.src_labels,1))
	    self.optimizer = tf.train.AdamOptimizer(0.001) 
	    self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

	    # summary op
	    loss_summary = tf.summary.scalar('classification_loss', self.loss)
	    fzy_accuracy_summary = tf.summary.scalar('fzy_accuracy', self.fzy_accuracy)
	    fx_src_accuracy_summary = tf.summary.scalar('fx_src_accuracy', self.fx_src_accuracy)
	    fx_trg_accuracy_summary = tf.summary.scalar('fx_trg_accuracy', self.fx_trg_accuracy)
	    self.summary_op = tf.summary.merge([loss_summary, fzy_accuracy_summary, fx_src_accuracy_summary, fx_trg_accuracy_summary])




