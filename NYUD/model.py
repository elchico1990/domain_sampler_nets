import tensorflow as tf
import tensorflow.contrib.slim as slim

# Issue #6064: slim does not import nets, must import manually
# import tensorflow.contrib.slim.nets as nets # implemented here to return fc layers

import numpy as np



class DSN(object):
    """Domain Sampler Network
    """
    def __init__(self, mode='train', learning_rate=0.0001):
        self.mode = mode
        self.learning_rate = learning_rate
	self.hidden_repr_size = 512
	self.no_classes = 19

    
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
		    net = slim.fully_connected(net, 2048, activation_fn = tf.nn.relu, scope='sgen_fc2')
		    net = slim.batch_norm(net, scope='sgen_bn2')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 4096, activation_fn = tf.nn.relu, scope='sgen_fc3')
		    net = slim.batch_norm(net, scope='sgen_bn3')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 4096, activation_fn = tf.nn.relu, scope='sgen_fc4')
		    net = slim.batch_norm(net, scope='sgen_bn4')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn = tf.tanh, scope='sgen_feat')
		    return net
		    
    def E(self, images, reuse=False, make_preds=False, is_training = False, scope='encoder'):
	
	#~ _mode = self.mode == 'eval_dsn'
	
	with tf.variable_scope('vgg_16', reuse=reuse):
	    # vgg16  as in https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim#working-example-specifying-the-vgg16-layers
	    with slim.arg_scope([slim.conv2d],
			  activation_fn=tf.nn.relu,
			  weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			  weights_regularizer=slim.l2_regularizer(0.0005)):
		#~ with tf.device('/gpu:0' if not _mode else '/cpu:0'):
		with tf.device('/gpu:0'):
		    net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		    net = slim.max_pool2d(net, [2, 2], scope='pool1')
		    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		    net = slim.max_pool2d(net, [2, 2], scope='pool2')
		    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		    net = slim.max_pool2d(net, [2, 2], scope='pool3')	
		    	
		#~ with tf.device('/gpu:1' if not _mode else '/cpu:0'):
		with tf.device('/gpu:1'):
		    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		    net = slim.max_pool2d(net, [2, 2], scope='pool4')
		    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		    net = slim.max_pool2d(net, [2, 2], scope='pool5')
		    net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
		    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
		    ## differs from vgg
		    net = slim.conv2d(net, self.hidden_repr_size , [1, 1], padding='VALID', activation_fn=tf.tanh, scope='fc7')
		    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
		    if (self.mode == 'pretrain' or self.mode == 'test' or make_preds):
			net = slim.conv2d(net, self.no_classes , [1,1], activation_fn=None, scope='fc8')
			
	return net
			    
    def D_e(self, inputs, y, reuse=False):
		
	inputs = tf.concat(axis=1, values=[tf.contrib.layers.flatten(inputs), tf.cast(y,tf.float32)])
	
	with tf.variable_scope('disc_e',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.zeros_initializer()):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train_sampler')):
                    
		    if self.mode == 'train_sampler':
			net = slim.fully_connected(inputs, 3, activation_fn = tf.nn.relu, scope='sdisc_fc1')
			#~ net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sdisc_fc2')
		    elif self.mode == 'train_dsn':
			net = slim.fully_connected(inputs, 1024, activation_fn = tf.nn.relu, scope='sdisc_fc1')
			net = slim.fully_connected(net, 2048, activation_fn = tf.nn.relu, scope='sdisc_fc2')
			#~ net = slim.fully_connected(net, 2048, activation_fn = tf.nn.relu, scope='sdisc_fc3')
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='sdisc_prob')
		    return net

    def build_model(self):
              
        if self.mode == 'pretrain' or self.mode == 'test' or self.mode == 'test_ensemble':
            
	    self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'source_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'target_images')
            self.src_labels = tf.placeholder(tf.int64, [None], 'source_labels')
            self.trg_labels = tf.placeholder(tf.int64, [None], 'target_labels')
	    self.keep_prob = tf.placeholder(tf.float32)
	    
	    self.src_logits = self.E(self.src_images, is_training = True)
		
	    self.src_pred = tf.argmax(tf.squeeze(self.src_logits), 1) #logits are [self.no_classes ,1,1,8], need to squeeze
            self.src_correct_pred = tf.equal(self.src_pred, self.src_labels) 
            self.src_accuracy = tf.reduce_mean(tf.cast(self.src_correct_pred, tf.float32))
		
            self.trg_logits = self.E(self.trg_images, is_training = False, reuse=True)
		
	    self.trg_pred = tf.argmax(tf.squeeze(self.trg_logits), 1) #logits are [self.no_classes ,1,1,8], need to squeeze
            self.trg_correct_pred = tf.equal(self.trg_pred, self.trg_labels)
            self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))
	    
	    t_vars = tf.trainable_variables()
	    
	    
	    train_vars = t_vars #[var for var in t_vars if 'fc_repr' in var.name] + [var for var in t_vars if 'fc8' in var.name]
	    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.src_logits,labels=tf.one_hot(self.src_labels,self.no_classes )))
	    gradients = tf.gradients(self.loss, train_vars)
	    gradients = list(zip(gradients, train_vars))
	    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
	    self.train_op = self.optimizer.apply_gradients(grads_and_vars=gradients)
	    
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            src_accuracy_summary = tf.summary.scalar('src_accuracy', self.src_accuracy)
            trg_accuracy_summary = tf.summary.scalar('trg_accuracy', self.trg_accuracy)
            self.summary_op = tf.summary.merge([loss_summary, src_accuracy_summary, trg_accuracy_summary])
	
	elif self.mode == 'train_sampler':
				
	    self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'svhn_images')
	    self.fx = tf.placeholder(tf.float32, [None, self.hidden_repr_size], 'features')
	    self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
	    self.labels = tf.placeholder(tf.int64, [None, self.no_classes ], 'labels_real')
	    try:
		self.dummy_fx = self.E(self.images)
	    except:
		self.dummy_fx = self.E(self.images, reuse=True)
			
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
            self.src_labels = tf.placeholder(tf.float32, [None, self.no_classes ], 'labels')
	    self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'src_images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'trg_images')
            
	    self.fx_src = self.E(self.src_images, is_training=False)  
            self.fx_trg = self.E(self.trg_images, reuse=True, is_training=False) 
	    
	    self.fzy = self.sampler_generator(self.src_noise,self.src_labels)

	elif self.mode == 'train_dsn':
	    
            self.src_noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.src_labels = tf.placeholder(tf.float32, [None, self.no_classes ], 'labels')
            self.src_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'mnist_images')
	    
	    self.trg_labels = self.E(self.trg_images, make_preds=True)
	    self.trg_labels = tf.one_hot(tf.argmax(self.trg_labels,1),self.no_classes )
	    
	    self.images = tf.concat(axis=0, values=[self.src_images, self.trg_images])
	    self.labels = tf.concat(axis=0, values=[self.src_labels,self.trg_labels])
	    
	    self.fzy = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
	    
	    self.fx = self.E(self.images, reuse=True)
	    
	    # E losses
	    
	    self.logits_E_real = self.D_e(self.fzy, self.src_labels)
	    self.logits_E_fake = self.D_e(self.fx, self.labels, reuse=True)
	    
	    self.DE_loss_real = tf.reduce_mean(tf.square(self.logits_E_real - tf.ones_like(self.logits_E_real)))
	    self.DE_loss_fake = tf.reduce_mean(tf.square(self.logits_E_fake - tf.zeros_like(self.logits_E_fake)))
	    
	    self.DE_loss = self.DE_loss_real + self.DE_loss_fake 
	    
	    self.E_loss = tf.reduce_mean(tf.square(self.logits_E_fake - tf.ones_like(self.logits_E_fake)))
	    
	    # Optimizers
	   
            
            t_vars = tf.trainable_variables()
            #~ E_vars = [var for var in t_vars if 'fc_repr' in var.name] + [var for var in t_vars if 'fc8' in var.name]
	    E_vars = [v for v in t_vars if np.all([s not in str(v.name) for s in ['encoder','sampler_generator','generator','disc_e','disc_g','source_train_op','training_op']])]
	    DE_vars = [var for var in t_vars if 'disc_e' in var.name]
            
	    E_gradients = tf.gradients(self.E_loss, E_vars)
	    E_gradients = list(zip(E_gradients, E_vars))
	    self.E_optimizer = tf.train.AdamOptimizer(self.learning_rate / 100.)
	    
	    DE_gradients = tf.gradients(self.DE_loss, DE_vars)
	    DE_gradients = list(zip(DE_gradients, DE_vars))
	    self.DE_optimizer = tf.train.AdamOptimizer(self.learning_rate / 100.)
	    
	    
	    self.E_train_op = self.E_optimizer.apply_gradients(grads_and_vars=E_gradients)
	    self.DE_train_op = self.DE_optimizer.apply_gradients(grads_and_vars=DE_gradients)
	    
	    
            
            # summary op
            E_loss_summary = tf.summary.scalar('E_loss', self.E_loss)
            DE_loss_summary = tf.summary.scalar('DE_loss', self.DE_loss)
            self.summary_op = tf.summary.merge([E_loss_summary, DE_loss_summary])
            

            for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var) 
		
		
		


		
		
		
		

		    
		
		
		
		



