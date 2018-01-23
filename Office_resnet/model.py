import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
# Issue #6064: slim does not import nets, must import manually
# import tensorflow.contrib.slim.nets as nets # implemented here to return fc layers

import numpy as np

from utils import lrelu


class DSN(object):
    """Domain Sampler Network
    """
    def __init__(self, mode='train', learning_rate=0.0001):
        self.mode = mode
        self.learning_rate = learning_rate
        self.hidden_repr_size = 48
        self.no_classes = 31
        self.noise_dim = 100

    
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
	    with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.0)):
		
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train_sampler')):
                    
		    net = slim.fully_connected(inputs, 1024, activation_fn = tf.nn.relu, scope='sgen_fc1')
		    net = slim.batch_norm(net, scope='sgen_bn1')
		    net = slim.dropout(net, 0.5)
		    res = slim.fully_connected(net, 1024 , activation_fn = tf.nn.relu, scope='sgen_fc2')
		    res = slim.batch_norm(res, scope='sgen_bn2')
		    res = slim.dropout(res, 0.5)
		    net = res+net
		    res = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc3')
		    res = slim.batch_norm(res, scope='sgen_bn3')
		    res = slim.dropout(res, 0.5)
		    net = res+net
		    res = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc4')
		    res = slim.batch_norm(res, scope='sgen_bn4')
		    res = slim.dropout(res, 0.5)
		    net = res+net
		    res = slim.fully_connected(net, 1024 , activation_fn = tf.nn.relu, scope='sgen_fc5')
		    res = slim.batch_norm(res, scope='sgen_bn5')
		    res = slim.dropout(res, 0.5)
		    net = res+net
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn = tf.nn.tanh, scope='sgen_feat')
		    return net
		    
		    
    def E(self, images, reuse=False, make_preds=False, is_training=False):
			      
	if self.mode=='features':
	    with tf.variable_scope('resnet_v1_50', reuse=reuse):
		with slim.arg_scope([slim.conv2d],
			      activation_fn=tf.nn.relu,
			      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			      weights_regularizer=slim.l2_regularizer(0.0005)):
		    
		    images = tf.reshape(images,[-1,1,1,self.hidden_repr_size])
		    return slim.conv2d(images, self.no_classes , [1,1], activation_fn=None, scope='_logits_')
	    
	    
	    
	with slim.arg_scope(resnet_v1.resnet_arg_scope()):
	    _, end_points = resnet_v1.resnet_v1_50(images, self.no_classes, is_training=is_training, reuse=reuse)
	
	with slim.arg_scope([slim.conv2d],
			  activation_fn=tf.nn.relu,
			  weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			  weights_regularizer=slim.l2_regularizer(0.0005)):
	    
	    net = end_points['resnet_v1_50/block4'] #last bottleneck before logits
	    
	    with tf.variable_scope('resnet_v1_50', reuse=reuse):
		net = slim.conv2d(net, self.hidden_repr_size , [7, 7], padding='VALID', activation_fn=tf.tanh, scope='f_repr')
		
		if (self.mode == 'pretrain' or self.mode == 'test' or make_preds):
			    #~ net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
		    net = slim.conv2d(net, self.no_classes , [1,1], activation_fn=None, scope='_logits_')
		
	    return net
	    
	    
    def D_e(self, inputs, y, reuse=False):
		
	inputs = tf.concat(axis=1, values=[tf.contrib.layers.flatten(inputs), tf.cast(y,tf.float32)])
	
	with tf.variable_scope('disc_e',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.zeros_initializer()):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=lrelu, is_training=(self.mode=='train_sampler')):
                    
		    if self.mode == 'train_sampler':
			net = slim.fully_connected(inputs,64, activation_fn = lrelu, scope='sdisc_fc1')
			#~ net = slim.fully_connected(net, 256, activation_fn = lrelu, scope='sdisc_fc2')
		    elif self.mode == 'train_dsn' or 'train_adda' in self.mode :
			net = slim.fully_connected(inputs, 1024, activation_fn = lrelu, scope='sdisc_fc1')
			net = slim.fully_connected(net, 1024, activation_fn = lrelu, scope='sdisc_fc2')##
			net = slim.fully_connected(net, 1024, activation_fn = lrelu, scope='sdisc_fc3')
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='sdisc_prob')
		    return net
		    

    def G(self, inputs, labels, reuse=False, do_reshape=False):
	
	labels = tf.reshape(labels, [-1, 1, 1, self.no_classes])
	
	if inputs.get_shape()[1] != 1:
	    inputs = tf.expand_dims(inputs, 1)
	    inputs = tf.expand_dims(inputs, 1)
	
	inputs = conv_concat(inputs, labels)
	
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train_dsn')):

                    net = slim.conv2d_transpose(inputs, 512, [7, 7], padding='VALID', scope='conv_transpose1')   # (batch_size, 7, 7, #)
                    net = slim.batch_norm(net, scope='bn1')
		    net = conv_concat(net, labels)
                    net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')  # (batch_size, 14, 14, #)
                    net = slim.batch_norm(net, scope='bn2')
                    net = conv_concat(net, labels)
		    net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')  # (batch_size, 28, 28, #)
                    net = slim.batch_norm(net, scope='bn3')
                    net = conv_concat(net, labels)
		    net = slim.conv2d_transpose(net, 64, [3, 3], scope='conv_transpose4')  # (batch_size, 56, 56, #)
                    net = slim.batch_norm(net, scope='bn4')
                    net = conv_concat(net, labels)
		    net = slim.conv2d_transpose(net, 32, [3, 3], scope='conv_transpose5')  # (batch_size, 112, 112, #)
                    net = slim.batch_norm(net, scope='bn5')
                    net = conv_concat(net, labels)
		    net = slim.conv2d_transpose(net, 3, [3, 3], activation_fn=tf.tanh, scope='conv_transpose6')   # (batch_size, 224, 224, 3)
		    return net
	
	
    def D_g(self, images, labels, reuse=False):
	
	labels = tf.reshape(labels, [-1, 1, 1, self.no_classes])
	
	images = conv_concat(images, labels)
	
        # images: (batch, 32, 32, 1)
        with tf.variable_scope('disc_g', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=lrelu, is_training=(self.mode=='train_dsn')):
                    
                    net = slim.conv2d(images, 128, [3, 3], scope='conv1')   # (batch_size, 14, 14 128)
                    net = slim.batch_norm(net, scope='bn1')
		    net = conv_concat(net, labels)
                    net = slim.conv2d(net, 256, [3, 3], scope='conv2')   # (batch_size, 7, 7, 256)
                    net = slim.batch_norm(net, scope='bn2')
		    net = conv_concat(net, labels)
                    net = slim.conv2d(net, 512, [3, 3], scope='conv3')   # (batch_size, 7, 7, 256)
                    net = slim.batch_norm(net, scope='bn3')
		    net = conv_concat(net, labels)
                    net = slim.flatten(net)
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='fc1')   # (batch_size, 1)
		    return net	    
    
    
    def build_model(self):
              
        if self.mode == 'pretrain' or self.mode == 'test':
            
	    self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'source_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'target_images')
            self.src_labels = tf.placeholder(tf.int64, [None], 'source_labels')
            self.trg_labels = tf.placeholder(tf.int64, [None], 'target_labels')
	    #~ self.keep_prob = tf.placeholder(tf.float32)
	    
	    self.src_logits = self.E(self.src_images, is_training = True)
	    #~ print self.src_logits.get_shape()
		
	    self.src_pred = tf.argmax(tf.squeeze(self.src_logits), 1) #logits are [self.no_classes ,1,1,classes], need to squeeze
            self.src_correct_pred = tf.equal(self.src_pred, self.src_labels) 
            self.src_accuracy = tf.reduce_mean(tf.cast(self.src_correct_pred, tf.float32))
		
            self.trg_logits = self.E(self.trg_images, is_training = False, reuse=True)
		
	    self.trg_pred = tf.argmax(tf.squeeze(self.trg_logits), 1) #logits are [self.no_classes ,1,1,classes], need to squeeze
            self.trg_correct_pred = tf.equal(self.trg_pred, self.trg_labels)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))
	    
	    t_vars = tf.trainable_variables()
	    
	    train_vars = t_vars#[var for var in t_vars if 'logits' in var.name]
	    #~ for v in train_vars:
		#~ print v
	    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.src_logits,labels=tf.one_hot(self.src_labels,self.no_classes )))
	    gradients = tf.gradients(self.loss, train_vars)
	    gradients = list(zip(gradients, train_vars))
	    #for some reason Adam leads to severe overfitting
	    #~ self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
	    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
	    self.train_op = self.optimizer.apply_gradients(grads_and_vars=gradients)
	    
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            src_accuracy_summary = tf.summary.scalar('src_accuracy', self.src_accuracy)
            trg_accuracy_summary = tf.summary.scalar('trg_accuracy', self.trg_accuracy)
            self.summary_op = tf.summary.merge([loss_summary, src_accuracy_summary, trg_accuracy_summary])
	
	elif self.mode == 'train_sampler':
				
	    self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'source_images')
	    self.fx = tf.placeholder(tf.float32, [None, self.hidden_repr_size], 'features')
	    self.noise = tf.placeholder(tf.float32, [None, self.noise_dim], 'noise')
	    self.labels = tf.placeholder(tf.int64, [None, self.no_classes ], 'labels_real')
	    try:
		self.dummy_fx = slim.flatten(self.E(self.images))
	    except:
		self.dummy_fx = slim.flatten(self.E(self.images, reuse=True))
			
	    self.fzy = self.sampler_generator(self.noise, self.labels)

	    self.logits_real = self.D_e(self.fx,self.labels, reuse=False) 
	    self.logits_fake = self.D_e(self.fzy,self.labels, reuse=True)
	    
	    self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = self.d_loss_real + self.d_loss_fake
	    
	    self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))
	    
	    #~ self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate/10.,beta1=0.5)
	    #~ self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate/10.,beta1=0.5)
	    self.d_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
	    self.g_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
	    
	    
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

	    #~ for var in tf.trainable_variables():
		#~ tf.summary.histogram(var.op.name, var)
		
	elif  'train_adda' in self.mode:
				
	    self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'source_images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'target_images')
	    self.src_fx = tf.placeholder(tf.float32, [None, self.hidden_repr_size], 'source_features')
	    self.src_labels = tf.placeholder(tf.float32, [None, self.no_classes ], 'source_lables')
	    
	    #~ self.trg_logits = tf.squeeeze(self.E(self.trg_images, make_preds=True))
	    ## squeeze gives problems since it forgets shape
	    self.trg_logits = tf.reshape(self.E(self.trg_images, make_preds=True), (-1,self.no_classes))
	    self.trg_labels = tf.one_hot(tf.argmax(self.trg_logits,1),self.no_classes )
	    
	     #################
	    #temporarily added just to print out test accuracy during dsn training
	    self.trg_pred = tf.argmax(self.trg_logits, 1) 
	    self.target_labels = tf.placeholder(tf.int64, [None], 'target_labels') #name different from inferrend labels
            self.trg_correct_pred = tf.equal(self.trg_pred, self.target_labels)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))
	    #################
	    
	    
	    if self.mode == 'train_adda_shared':
		self.images = tf.concat(axis=0, values=[self.src_images, self.trg_images])
		self.labels = tf.concat(axis=0, values=[self.src_labels,self.trg_labels])
	    elif self.mode == 'train_adda':
		self.images = self.trg_images
		self.labels = self.trg_labels

	    
	    self.shared_fx = slim.flatten(self.E(self.images, reuse=True))

	    try:
		self.dummy_fx = slim.flatten(self.E(self.src_images))
	    except:
		self.dummy_fx = slim.flatten(self.E(self.src_images, reuse=True))


			
	    self.logits_real = self.D_e(self.src_fx,self.src_labels, reuse=False) 
	    self.logits_fake = self.D_e(self.shared_fx,self.labels, reuse=True)
	    
	    self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = self.d_loss_real + self.d_loss_fake
	    
	    self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))
	    
	    self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate/100, beta1=0.5)
	    self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate/100, beta1=0.5)
	    

	    
	    t_vars = tf.trainable_variables()
	    d_vars = [var for var in t_vars if 'disc_e' in var.name]
	    #~ g_vars = [var for var in t_vars if 'block4' not in var.name]
	    g_vars = [var for var in t_vars if 'resnet_v1_50' in var.name]
	    
	    # train op
	    with tf.variable_scope('source_train_op',reuse=False):
		self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=d_vars)
		self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=g_vars)
	    
	    # summary op
	    d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
	    g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
	    self.summary_op = tf.summary.merge([d_loss_summary, g_loss_summary])

	    #~ for var in tf.trainable_variables():
		#~ tf.summary.histogram(var.op.name, var)
        
	elif self.mode == 'eval_dsn':
            self.src_noise = tf.placeholder(tf.float32, [None, self.noise_dim], 'noise')
            self.src_labels = tf.placeholder(tf.float32, [None, self.no_classes ], 'labels')
	    self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'src_images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'trg_images')
            
	    self.fx_src = self.E(self.src_images, is_training=False)  
            self.fx_trg = self.E(self.trg_images, reuse=True, is_training=False) 
	    
	    self.fzy = self.sampler_generator(self.src_noise,self.src_labels)

	elif self.mode == 'train_dsn':
	    
            self.src_noise = tf.placeholder(tf.float32, [None, self.noise_dim], 'noise')
            self.src_labels = tf.placeholder(tf.float32, [None, self.no_classes ], 'labels')
            self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'source_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'target_images')
	    
	    self.trg_logits = tf.squeeze(self.E(self.trg_images, make_preds=True))
	    self.trg_labels = tf.one_hot(tf.argmax(self.trg_logits,1),self.no_classes )
	    
	    #################
	    #temporarily added just to print out test accuracy during dsn training
	    self.trg_pred = tf.argmax(tf.squeeze(self.trg_logits), 1) #logits are [self.no_classes ,1,1,8], need to squeeze
	    self.target_labels = tf.placeholder(tf.int64, [None], 'target_labels')
            self.trg_correct_pred = tf.equal(self.trg_pred, self.target_labels)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))
	    #################
	    
	    self.images = tf.concat(axis=0, values=[self.src_images, self.trg_images])
	    self.labels = tf.concat(axis=0, values=[self.src_labels,self.trg_labels])
	    
	    self.fzy = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
	    
	    self.fx = self.E(self.images, reuse=True)
	    
	    # E losses
	    # Sampler is fixed. We want to finetune E so that its features are similar to those generated by sampler_generator
	    # (Note that D_e is different from the one ised to train sampler_generator)
	    self.logits_E_real = self.D_e(self.fzy, self.src_labels)
	    self.logits_E_fake = self.D_e(self.fx, self.labels, reuse=True)
	    
	    self.DE_loss_real = tf.reduce_mean(tf.square(self.logits_E_real - tf.ones_like(self.logits_E_real)))
	    self.DE_loss_fake = tf.reduce_mean(tf.square(self.logits_E_fake - tf.zeros_like(self.logits_E_fake)))
	    
	    self.DE_loss = self.DE_loss_real + self.DE_loss_fake 
	    
	    self.E_loss = tf.reduce_mean(tf.square(self.logits_E_fake - tf.ones_like(self.logits_E_fake)))
           
	    # Optimizers
	    
            #~ self.DE_optimizer = tf.train.AdamOptimizer(self.learning_rate / 100.)
            #~ self.E_optimizer = tf.train.AdamOptimizer(self.learning_rate / 100.)
	    self.d_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
	    self.g_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            
            
            t_vars = tf.trainable_variables()
            E_vars = [var for var in t_vars if 'resnet_v1_50' in var.name]
            DE_vars = [var for var in t_vars if 'disc_e' in var.name]
            
            # train op
            with tf.variable_scope('training_op',reuse=False):
                self.E_train_op = slim.learning.create_train_op(self.E_loss, self.E_optimizer, variables_to_train=E_vars)
                self.DE_train_op = slim.learning.create_train_op(self.DE_loss, self.DE_optimizer, variables_to_train=DE_vars)
		
            # summary op
            E_loss_summary = tf.summary.scalar('E_loss', self.E_loss)
            DE_loss_summary = tf.summary.scalar('DE_loss', self.DE_loss)

            self.summary_op = tf.summary.merge([E_loss_summary, DE_loss_summary])
            
            #~ for var in tf.trainable_variables():
		#~ tf.summary.histogram(var.op.name, var)
	
	
	elif self.mode == 'features':
	    
            #~ self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'source_images')
	    #~ self.fx = self.E(self.images)
	    

	    self.noise = tf.placeholder(tf.float32, [None, self.noise_dim], 'noise')
	    self.labels = tf.placeholder(tf.int64, [None, self.no_classes ], 'labels_real')
	    
		
	    self.fzy = self.sampler_generator(self.noise, self.labels) 		
	    self.inferred_labels = tf.argmax(slim.flatten(self.E(self.fzy)),1)
	   

		
		
		


		
		
		
		

		    
		
		
		
		



