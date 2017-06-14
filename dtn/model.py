import tensorflow as tf
import tensorflow.contrib.slim as slim
import cPickle


class DSN(object):
    """Domain Transfer Network
    """
    def __init__(self, mode='train', learning_rate=0.0003):
        self.mode = mode
        self.learning_rate = learning_rate
	self.hidden_repr_size = 256
	
	#~ if self.mode in ['train_dsn','eval_dsn']:
	    #~ with open('./model/min_max_file.pkl','r') as f:
		#~ self.featsMin, self.featsMax = cPickle.load(f)
	    
    def sampler_discriminator(self, x, y, reuse=False):
	
	#~ x = tf.reshape(x,[-1,128])
	inputs = tf.concat(axis=1, values=[x, y])
	#~ inputs = x
	    
	with tf.variable_scope('sampler_discriminator',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.zeros_initializer()):
		#~ net = slim.flatten(inputs)
		net = slim.fully_connected(inputs, 1024, activation_fn = tf.nn.relu, scope='sdisc_fc1')
		net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sdisc_fc2')
		net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='sdisc_prob')
		return net

    def sampler_generator(self, z, y, reuse=False):
	'''
	Takes in input noise and labels, and
	generates f_z, which is handled by the 
	net as f(x) was handled. If labels is 
	None, the noise samples are partitioned
	in equal ratios.  
	'''
	
	inputs = tf.concat(axis=1, values=[z, y])
	#~ inputs = z
	
	with tf.variable_scope('sampler_generator', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.zeros_initializer()):
		net = slim.fully_connected(inputs, 1024, activation_fn = tf.nn.relu, scope='sgen_fc1')
		net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc2')
		net = slim.fully_connected(net, self.hidden_repr_size, activation_fn = tf.nn.sigmoid, scope='sgen_feat')
		return net
        
    def content_extractor(self, images, reuse=False, class_prob=False):
        # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        if images.get_shape()[3] == 1:
            # For mnist dataset, replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)
        
        with tf.variable_scope('content_extractor', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train' or self.mode=='pretrain')):
                    
                    net = slim.conv2d(images, 64, [3, 3], scope='conv1')   # (batch_size, 16, 16, 64)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2')     # (batch_size, 8, 8, 128)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3')     # (batch_size, 4, 4, 256)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, self.hidden_repr_size, [4, 4], padding='VALID', scope='conv4')   # (batch_size, 1, 1, 128)
                    net = slim.batch_norm(net, scope='bn4')
		    net = slim.flatten(net)
                    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn = tf.nn.sigmoid, scope='fc1')
		    
		    if self.mode == 'pretrain':
                        net = slim.fully_connected(net, 10, activation_fn = tf.nn.sigmoid, scope='out')
		    return net
        
    #~ def content_extractor(self, images, reuse=False, class_prob=False):
        #~ # images: (batch, 32, 32, 3) or (batch, 32, 32, 1)
        
        #~ if images.get_shape()[3] == 1:
            #~ # For mnist dataset, replicate the gray scale image 3 times.
            #~ images = tf.image.grayscale_to_rgb(images)
        
        #~ with tf.variable_scope('content_extractor', reuse=reuse):
            #~ with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                 #~ stride=2,  weights_initializer=tf.contrib.layers.xavier_initializer()):
	    
		#~ net = slim.conv2d(images, 64, [3, 3], scope='conv1')   # (batch_size, 16, 16, 64)
		#~ net = slim.avg_pool2d(net, [3, 3], scope='pool1')
		#~ net = slim.conv2d(net, 128, [3, 3], scope='conv2')     # (batch_size, 8, 8, 128)
		#~ net = slim.avg_pool2d(net, [3, 3], scope='pool22')
		#~ net = slim.flatten(net)
		#~ net = slim.fully_connected(net, 512, activation_fn = tf.nn.relu, scope='fc1')
		#~ net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.sigmoid, scope='fc2')
		
		#~ if self.mode == 'pretrain':
		    #~ net = slim.fully_connected(net, 10, activation_fn=tf.sigmoid, scope='out')
		    
		#~ if class_prob:
		    #~ net = slim.fully_connected(net, 10, scope='out')
		    
		#~ return net
	    
    def generator(self, inputs, reuse=False, from_samples=False):
        # inputs: (batch, 1, 1, 128)
	
	#~ if from_samples:
	    #~ inputs = tf.multiply(inputs,(self.featsMax - self.featsMin))
	    #~ inputs = tf.add(inputs,self.featsMin)
	    
	if inputs.get_shape()[1] != 1:
	    inputs = tf.expand_dims(inputs, 1)
	    inputs = tf.expand_dims(inputs, 1)
	
        with tf.variable_scope('generator', reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', activation_fn=None,           
                                 stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
                                     activation_fn=tf.nn.relu, is_training=(self.mode=='train')):

                    net = slim.conv2d_transpose(inputs, 512, [4, 4], padding='VALID', scope='conv_transpose1')   # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')  # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d_transpose(net, 1, [3, 3], activation_fn=tf.nn.tanh, scope='conv_transpose4')   # (batch_size, 32, 32, 1)
                    return net
    
    def discriminator(self, images, reuse=False):
        # images: (batch, 32, 32, 1)
        with tf.variable_scope('discriminator', reuse=reuse):
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
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='fc1')   # (batch_size, 1)
                    return net
                
    def build_model(self):
        
        if self.mode == 'pretrain':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)
            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # loss and train op
            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
            
            # summary op
            loss_summary = tf.summary.scalar('classification_loss', self.loss)
            accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])
	
	elif self.mode == 'train_sampler':
				
	    self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
	    self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
	    self.labels = tf.placeholder(tf.float32, [None, 10], 'labels')
	    self.fx = tf.placeholder(tf.float32, [None, self.hidden_repr_size], 'feat')
	    
	    self.fx_ext = self.content_extractor(self.images)
	    self.fzy = self.sampler_generator(self.noise, self.labels) 

	    self.logits_real = self.sampler_discriminator(self.fx,self.labels) 
	    self.logits_fake = self.sampler_discriminator(self.fzy,self.labels, reuse=True) 
	    
	    #~ d_loss_real = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(self.logits_real,tf.ones_like(self.logits_real)))
	    #~ d_loss_fake = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(self.logits_fake,tf.zeros_like(self.logits_fake)))
    
	    d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = d_loss_real + d_loss_fake
	    #~ self.g_loss = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(self.logits_fake,tf.ones_like(self.logits_fake)))
	    
	    self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))

	    self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
	    self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate)
	    
	    t_vars = tf.trainable_variables()
	    d_vars = [var for var in t_vars if 'sampler_discriminator' in var.name]
	    g_vars = [var for var in t_vars if 'sampler_generator' in var.name]
	    f_vars = [var for var in t_vars if 'content_extractor' in var.name]
	    
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
        
	elif self.mode == 'eval':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')

            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)
        
	elif self.mode == 'eval_dsn':
            self.src_noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.src_labels = tf.placeholder(tf.float32, [None, 10], 'labels')
            
            # source domain (svhn to mnist)
            self.fx = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
	    self.sampled_images = self.generator(self.fx,from_samples=True)

        elif self.mode == 'train':
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
            
            # source domain (svhn to mnist)
            self.fx = self.content_extractor(self.src_images)
            self.fake_images = self.generator(self.fx)
            self.logits = self.discriminator(self.fake_images)
            self.fgfx = self.content_extractor(self.fake_images, reuse=True)

            # loss
            self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.zeros_like(self.logits))
            self.g_loss_src = slim.losses.sigmoid_cross_entropy(self.logits, tf.ones_like(self.logits))
            self.f_loss_src = tf.reduce_mean(tf.square(self.fx - self.fgfx)) * 15.0
            
            # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'content_extractor' in var.name]
            
            # train op
            with tf.variable_scope('source_train_op',reuse=False):
                self.d_train_op_src = slim.learning.create_train_op(self.d_loss_src, self.d_optimizer_src, variables_to_train=d_vars)
                self.g_train_op_src = slim.learning.create_train_op(self.g_loss_src, self.g_optimizer_src, variables_to_train=g_vars)
                self.f_train_op_src = slim.learning.create_train_op(self.f_loss_src, self.f_optimizer_src, variables_to_train=g_vars)
            
            # summary op
            d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            origin_images_summary = tf.summary.image('src_origin_images', self.src_images)
            sampled_images_summary = tf.summary.image('src_sampled_images', self.fake_images)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary, 
                                                    f_loss_src_summary, origin_images_summary, 
                                                    sampled_images_summary])
            
            # target domain (mnist)
            self.fx = self.content_extractor(self.trg_images, reuse=True)
            self.reconst_images = self.generator(self.fx, reuse=True)
            self.logits_fake = self.discriminator(self.reconst_images, reuse=True)
            self.logits_real = self.discriminator(self.trg_images, reuse=True)
            
            # loss
            self.d_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            self.d_loss_real_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg
            self.g_loss_fake_trg = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))
            self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images)) * 15.0
            self.g_loss_trg = self.g_loss_fake_trg + self.g_loss_const_trg
            
            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            with tf.variable_scope('target_train_op',reuse=False):
                self.d_train_op_trg = slim.learning.create_train_op(self.d_loss_trg, self.d_optimizer_trg, variables_to_train=d_vars)
                self.g_train_op_trg = slim.learning.create_train_op(self.g_loss_trg, self.g_optimizer_trg, variables_to_train=g_vars)
            
            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)
            origin_images_summary = tf.summary.image('trg_origin_images', self.trg_images)
            sampled_images_summary = tf.summary.image('trg_reconstructed_images', self.reconst_images)
            self.summary_op_trg = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary, 
                                                    d_loss_fake_trg_summary, d_loss_real_trg_summary,
                                                    g_loss_fake_trg_summary, g_loss_const_trg_summary,
                                                    origin_images_summary, sampled_images_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
	
	elif self.mode == 'train_dsn':
            self.src_noise = tf.placeholder(tf.float32, [None, 100], 'noise')
            self.src_labels = tf.placeholder(tf.float32, [None, 10], 'labels')
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'mnist_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
	    
            # source domain (svhn to mnist)
            self.fx = self.sampler_generator(self.src_noise,self.src_labels) # instead of extracting the hidden representation from a src image, 
	    self.fake_images = self.generator(self.fx,from_samples=True)
            self.logits = self.discriminator(self.fake_images)
            self.fgfx = self.content_extractor(self.fake_images)

	    self.orig_fx = self.content_extractor(self.src_images, reuse=True)

	    #~ self.pred_src_labels = self.content_extractor(self.fake_images, class_prob=True)

            # loss
            self.d_loss_src = tf.reduce_mean(tf.square(self.logits - tf.zeros_like(self.logits)))
            self.g_loss_src = tf.reduce_mean(tf.square(self.logits - tf.ones_like(self.logits)))
            self.f_loss_src = tf.reduce_mean(tf.square(self.fx - self.fgfx)) * 15.0
            #~ self.l_loss_src = slim.losses.sigmoid_cross_entropy(self.pred_src_labels, self.src_labels))
            
	    # optimizer
            self.d_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            self.f_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
            #~ self.l_optimizer_src = tf.train.AdamOptimizer(self.learning_rate)
	    
            
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            f_vars = [var for var in t_vars if 'content_extractor' in var.name]
            
            # train op
            with tf.variable_scope('source_train_op',reuse=False):
                self.d_train_op_src = slim.learning.create_train_op(self.d_loss_src, self.d_optimizer_src, variables_to_train=d_vars)
                self.g_train_op_src = slim.learning.create_train_op(self.g_loss_src, self.g_optimizer_src, variables_to_train=g_vars)
                self.f_train_op_src = slim.learning.create_train_op(self.f_loss_src, self.f_optimizer_src, variables_to_train=g_vars)
            
            # summary op
            d_loss_src_summary = tf.summary.scalar('src_d_loss', self.d_loss_src)
            g_loss_src_summary = tf.summary.scalar('src_g_loss', self.g_loss_src)
            f_loss_src_summary = tf.summary.scalar('src_f_loss', self.f_loss_src)
            sampled_images_summary = tf.summary.image('src_sampled_images', self.fake_images)
            self.summary_op_src = tf.summary.merge([d_loss_src_summary, g_loss_src_summary, 
                                                    f_loss_src_summary, sampled_images_summary])
            
            # target domain (mnist)
            self.fx_trg = self.content_extractor(self.trg_images, reuse=True)
            self.reconst_images_trg = self.generator(self.fx_trg, reuse=True)
            self.logits_fake_trg = self.discriminator(self.reconst_images_trg, reuse=True)
            self.logits_real_trg = self.discriminator(self.trg_images, reuse=True)
            
            # loss
            self.d_loss_fake_trg = tf.reduce_mean(tf.square(self.logits_fake_trg - tf.zeros_like(self.logits_fake_trg)))
            self.d_loss_real_trg = tf.reduce_mean(tf.square(self.logits_real_trg - tf.ones_like(self.logits_real_trg)))
            self.d_loss_trg = self.d_loss_fake_trg + self.d_loss_real_trg
            self.g_loss_fake_trg = tf.reduce_mean(tf.square(self.logits_fake_trg - tf.ones_like(self.logits_fake_trg)))
            self.g_loss_const_trg = tf.reduce_mean(tf.square(self.trg_images - self.reconst_images_trg))
            self.g_loss_trg = self.g_loss_fake_trg 
            
            # optimizer
            self.d_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_trg = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer_const_trg = tf.train.AdamOptimizer(self.learning_rate)

            # train op
            with tf.variable_scope('target_train_op',reuse=False):
                self.d_train_op_trg = slim.learning.create_train_op(self.d_loss_trg, self.d_optimizer_trg, variables_to_train=d_vars)
                self.g_train_op_trg = slim.learning.create_train_op(self.g_loss_trg, self.g_optimizer_trg, variables_to_train=g_vars)
                self.g_train_op_const_trg = slim.learning.create_train_op(self.g_loss_const_trg, self.g_optimizer_const_trg, variables_to_train=g_vars)
            
            # summary op
            d_loss_fake_trg_summary = tf.summary.scalar('trg_d_loss_fake', self.d_loss_fake_trg)
            d_loss_real_trg_summary = tf.summary.scalar('trg_d_loss_real', self.d_loss_real_trg)
            d_loss_trg_summary = tf.summary.scalar('trg_d_loss', self.d_loss_trg)
            g_loss_fake_trg_summary = tf.summary.scalar('trg_g_loss_fake', self.g_loss_fake_trg)
            g_loss_const_trg_summary = tf.summary.scalar('trg_g_loss_const', self.g_loss_const_trg)
            g_loss_trg_summary = tf.summary.scalar('trg_g_loss', self.g_loss_trg)
            origin_images_summary = tf.summary.image('trg_origin_images', self.trg_images)
            sampled_images_summary = tf.summary.image('trg_reconstructed_images', self.reconst_images_trg)
            self.summary_op_trg = tf.summary.merge([d_loss_trg_summary, g_loss_trg_summary, 
                                                    d_loss_fake_trg_summary, d_loss_real_trg_summary,
                                                    g_loss_fake_trg_summary, g_loss_const_trg_summary,
                                                    origin_images_summary, sampled_images_summary])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    

	    
	    



