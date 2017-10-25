import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt

import cPickle

import urllib2

slim = tf.contrib.slim

import vgg
import vgg_preprocessing
from vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)

from load_synthia import load_synthia
import utils

class DSN(object):
    
    def __init__(self, seq_name, no_classes=13):

	self.seq_name = seq_name
	self.no_classes = no_classes
	self.log_dir = './logs'
	self.vgg_checkpoint_path = './vgg_16.ckpt'
	
    def vgg_encoding(self, processed_images, is_training_placeholder, reuse=False): 
		
	with slim.arg_scope(vgg.vgg_arg_scope()):

	    fc7 = vgg.vgg_16(processed_images,
				num_classes=self.no_classes,
				is_training=is_training_placeholder,
				spatial_squeeze=False,
				fc_conv_padding='VALID',
				reuse=reuse,
				return_fc7=True)
				
	    return fc7
    
    def semantic_extractor(self, vgg_output):
	
	with tf.device('/gpu:3'):
	
	    with tf.variable_scope('semantic_extractor'):
		
		#~ # fc7 is (batch_size, 8, 8, 4096)
    #~ 
		#~ net_tmp = slim.conv2d(vgg_output, 256, [8, 8], scope='conv_plus_1', stride=1, padding='VALID', activation_fn=tf.nn.tanh) #from (batch_size, 8, 8, 4096) to (batch_size, 1, 1, 1024) 
    #~ 
		#~ fc_bottleneck = tf.squeeze(net_tmp) # (batch_size, 1024)

		net = slim.conv2d_transpose(vgg_output, 256, [7, 7], padding='VALID', scope='dec0')                # (batch_size, 7, 7, 512)
		net = slim.conv2d_transpose(net, 256, [3, 3], stride=2, padding='SAME', scope='dec1')           # (batch_size, 14, 14, 512)
				
		net = slim.conv2d_transpose(net, 256, [3, 3],stride=2,  padding='SAME', scope='dec2')   	     # (batch_size, 28, 28, 512)
		
		net = slim.conv2d_transpose(net, 256, [3, 3],stride=1,  padding='SAME', scope='dec21') 	     # (batch_size, 28, 28, 512)
		net = slim.conv2d_transpose(net, 256, [3, 3],stride=1,  padding='SAME', scope='dec22')	     # (batch_size, 28, 28, 512)
		net = slim.conv2d_transpose(net, 256, [3, 3],stride=1,  padding='SAME', scope='dec23')	     # (batch_size, 28, 28, 512)
		
		net = slim.conv2d_transpose(net, 256, [3, 3],stride=2,  padding='SAME', scope='dec3')          # (batch_size, 56, 56, 512)
		
		net = slim.conv2d_transpose(net, 256, [3, 3],stride=1,  padding='SAME', scope='dec31')	     # (batch_size, 56, 56, 512)
		net = slim.conv2d_transpose(net, 256, [3, 3],stride=1,  padding='SAME', scope='dec32')	     # (batch_size, 56, 56, 512)
		 
		net = slim.conv2d_transpose(net, 128, [3, 3],stride=1,  padding='SAME', scope='dec33')	     # (batch_size, 56, 56, 256)
		
		net = slim.conv2d_transpose(net, 128, [3, 3],stride=2,  padding='SAME', scope='dec4') 	     # (batch_size, 112, 112, 256)
		
		net9 = slim.conv2d_transpose(net, 128, [3, 3],stride=1,  padding='SAME', scope='dec41')	     # (batch_size, 112, 112, 256)
		net = slim.conv2d_transpose(net, 128, [3, 3],stride=1,  padding='SAME', scope='dec42')	     # (batch_size, 112, 112, 256)
		
		net = slim.conv2d_transpose(net, 64, [3, 3],stride=1,  padding='SAME', scope='dec43')       # (batch_size, 112, 112, 128)
		
		net = slim.conv2d_transpose(net, 64, [3, 3],stride=2,  padding='SAME', scope='dec5')        # (batch_size, 224, 224, 128)
		
		#~ net13 = slim.conv2d_transpose(net12, 128, [3, 3],stride=1,  padding='SAME', scope='dec51')       # (batch_size, 224, 224, 128)
		
		#~ net14 = slim.conv2d_transpose(net13, 64, [3, 3],stride=1,  padding='SAME', scope='dec52')        # (batch_size, 224, 224, 64)
		
		#~ net = slim.conv2d_transpose(net, 64, [3, 3],stride=2, padding='SAME', scope='dec6')          # (batch_size, 448, 448, 64)
		
		#~ net16 = slim.conv2d_transpose(net15, 64, [3, 3],stride=1, padding='SAME', scope='dec61')         # (batch_size, 448, 448, 64)
		#~ net17 = slim.conv2d_transpose(net16, 64, [3, 3],stride=1, padding='SAME', scope='dec62')         # (batch_size, 448,448, 64)
		
		logits = slim.conv2d(net, 13, [1, 1], scope='output')  				             # (batch_size, 448, 448, 13)
		
		return logits
	
    def feature_generator(self, noise, reuse=False, is_training=True):
    
	'''
	Takes in input noise, and generates 
	f_z, which is handled by the net as 
	f(x) was handled.  
	'''
    
	with tf.variable_scope('feature_generator', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.0)):
		
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
				    activation_fn=tf.nn.relu, is_training=is_training):
		    
		    net = slim.fully_connected(noise, 1024, activation_fn = tf.nn.relu, scope='sgen_fc1')
		    net = slim.batch_norm(net, scope='sgen_bn1')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 1024, activation_fn = tf.nn.relu, scope='sgen_fc2')
		    net = slim.batch_norm(net, scope='sgen_bn2')
		    net = slim.dropout(net, 0.5)
		    net = slim.fully_connected(net, 128, activation_fn = tf.tanh, scope='sgen_feat')
		    return net
		
    def feature_discriminator(self, inputs, reuse=False, is_training=True):

	with tf.variable_scope('feature_discriminator',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected],weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.0)):
		with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True, 
				    activation_fn=tf.nn.relu, is_training=is_training):
		    
		    net = slim.fully_connected(inputs, 128, activation_fn = tf.nn.relu, scope='sdisc_fc1')
		    net = slim.fully_connected(net,1,activation_fn=tf.sigmoid,scope='sdisc_prob')
		    return net
	    
    def build_model(self, mode='pretrain'):
	
	if mode=='train_semantic_extractor':
	
	    self.images = tf.placeholder(tf.float32, [None, 224 * 1,224 * 1, 3], 'images')
	    self.annotations = tf.placeholder(tf.float32, [None, 224 * 1,224 * 1, 1], 'annotations')
	    self.is_training_placeholder = tf.placeholder(tf.bool)

	    labels_tensors = [tf.to_float(tf.equal(self.annotations, i)) for i in range(self.no_classes)]

	    try:
		combined_mask = tf.concat(axis=3, values = labels_tensors)
	    except:
		combined_mask = tf.concat(3,labels_tensors)
		
	    flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, self.no_classes))

	    image_float = tf.to_float(self.images, name='ToFloat')

	    processed_images = tf.subtract(image_float, tf.constant([_R_MEAN, _G_MEAN, _B_MEAN]))
	    
	    # extracting VGG-16 representation, up to the (N-1) layer
	    
	    self.vgg_output = self.vgg_encoding(processed_images, self.is_training_placeholder)
	    self.vgg_output_flat = tf.squeeze(self.vgg_output)
	    
	    vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])
	    vgg_except_fc8_weights = slim.get_variables_to_restore(exclude= ['vgg_16/fc7','vgg_16/fc8'])
	    
	    # extracting semantic representation
	    
	    logits = self.semantic_extractor(self.vgg_output)
	    
	    flat_logits = tf.reshape(tensor=logits, shape=(-1, self.no_classes))

	    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
								      labels=flat_labels)

	    self.cross_entropy_sum = tf.reduce_mean(cross_entropies) # ORIGINAL WAS reduce_sum, CHECK PAPERS !!!

	    self.pred = tf.argmax(logits, dimension=3)

	    self.probabilities = tf.nn.softmax(logits)

	    # Optimizers

	    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

	    # no re-training of VGG-16 variables

	    t_vars = tf.trainable_variables()
	    self.train_vars = [var for var in t_vars if ('vgg_16' not in var.name) or ('fc7' in var.name)]

	    # train op
	    with tf.variable_scope('training_op',reuse=False):
		self.train_op = slim.learning.create_train_op(self.cross_entropy_sum, optimizer, variables_to_train=self.train_vars)

	    tf.summary.scalar('cross_entropy_loss', self.cross_entropy_sum)

	    self.merged_summary_op = tf.summary.merge_all()


	    # necessary to load VGG-16 weights

	    self.read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
		self.vgg_checkpoint_path,
		vgg_except_fc8_weights)

	    self.vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)
	    	
	if mode=='train_feature_generator':
	
	    self.fx = tf.placeholder(tf.float32, [None, 128], 'images')
	    self.noise = tf.placeholder(tf.float32, [None, 100], 'noise')
			
	    self.fzy = self.feature_generator(self.noise, is_training=True) 

	    self.logits_real = self.feature_discriminator(self.fx, reuse=False) 
	    self.logits_fake = self.feature_discriminator(self.fzy, reuse=True)
	    
	    self.d_loss_real = tf.reduce_mean(tf.square(self.logits_real - tf.ones_like(self.logits_real)))
	    self.d_loss_fake = tf.reduce_mean(tf.square(self.logits_fake - tf.zeros_like(self.logits_fake)))
	    
	    self.d_loss = self.d_loss_real + self.d_loss_fake
	    
	    self.g_loss = tf.reduce_mean(tf.square(self.logits_fake - tf.ones_like(self.logits_fake)))
	    
	    self.d_optimizer = tf.train.AdamOptimizer(0.0001)
	    self.g_optimizer = tf.train.AdamOptimizer(0.0001)
	    
	    t_vars = tf.trainable_variables()
	    d_vars = [var for var in t_vars if 'feature_discriminator' in var.name]
	    g_vars = [var for var in t_vars if 'feature_generator' in var.name]
	    
	    # train op
	    with tf.variable_scope('source_train_op',reuse=False):
		self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=d_vars)
		self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=g_vars)
	    
	    # summary op
	    d_loss_summary = tf.summary.scalar('feature_discriminator_loss', self.d_loss)
	    g_loss_summary = tf.summary.scalar('feature_generator_loss', self.g_loss)
	    self.summary_op = tf.summary.merge([d_loss_summary, g_loss_summary])

	    for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)




####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################


    def train_semantic_extractor(self):

	self.build_model('train_semantic_extractor')

	summary_string_writer = tf.summary.FileWriter(model.log_dir)

	config = tf.ConfigProto(device_count = {'GPU': 0})

        images, annotations = load_synthia(self.seq_name, no_elements=900)

	with tf.Session() as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    model.read_vgg_weights_except_fc8_func(sess)
	    sess.run(model.vgg_fc8_weights_initializer)
     
	    #~ # Run the initializers.
	    #~ sess.run(tf.global_variables_initializer())
	    #~ model.read_vgg_weights_except_fc8_func(sess)
	    #~ sess.run(model.vgg_fc8_weights_initializer)
	    #~ variables_to_restore = [i for i in slim.get_model_variables() if ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, './experiments/'+self.seq_name+'/model/segm_model')
	    #~ 
	    saver = tf.train.Saver(model.train_vars)
	    
	    feed_dict = {model.images: images,
			 model.annotations: annotations,
			 model.is_training_placeholder: False}

	    EPOCHS = 1000
	    BATCH_SIZE = 1

	    for e in range(EPOCHS):
		
		losses = []
		
		print e
		
		for n, start, end in zip(range(len(images)), range(0,len(images),BATCH_SIZE), range(BATCH_SIZE,len(images),BATCH_SIZE)):
			    
		    feed_dict = {model.images: images[start:end], model.annotations: annotations[start:end], model.is_training_placeholder: True}

		    loss, summary_string = sess.run([model.cross_entropy_sum, model.merged_summary_op], feed_dict=feed_dict)

		    sess.run(model.train_op, feed_dict=feed_dict)

		    summary_string_writer.add_summary(summary_string, e)

		    
		    if n%10==0:
			print e,'-',n
			losses.append(loss)
			print("Current Average Loss: " + str(np.array(losses).mean()))
		pred_np, probabilities_np = sess.run([model.pred, model.probabilities], feed_dict={model.images: images[1:2], model.annotations: annotations[1:2], model.is_training_placeholder: False})
		plt.imsave('./experiments/'+self.seq_name+'/images/'+str(e)+'.png', np.squeeze(pred_np))	    
		saver.save(sess, './experiments/'+self.seq_name+'/model/segm_model')

	    summary_string_writer.close()
	    
    def eval_semantic_extractor(self, seq_2_name):
	
	self.build_model('train_semantic_extractor')

	summary_string_writer = tf.summary.FileWriter(model.log_dir)

	config = tf.ConfigProto(device_count = {'GPU': 0})


	source_images, source_annotations = load_synthia(self.seq_name, no_elements=900)
	target_images, target_annotations = load_synthia(seq_2_name, no_elements=900)
		     
	source_features = np.zeros((len(source_images),128))
	target_features = np.zeros((len(target_images),128))
	source_losses = np.zeros((len(source_images), 1))
	
	source_preds = np.zeros((len(source_images),224,224))
	target_preds = np.zeros((len(target_images),224,224))
	target_losses = np.zeros((len(target_images), 1))
	

	with tf.Session() as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    self.read_vgg_weights_except_fc8_func(sess)
	    sess.run(self.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, './experiments/'+self.seq_name+'/model/segm_model')
	    
	    saver = tf.train.Saver(self.train_vars)

	    print 'Evaluating SOURCE - ' + self.seq_name
	    
	    for n, image, annotation in zip(range(len(source_images)), source_images, source_annotations):
		
		if n%100==0:
		    print n 
		feed_dict = {self.images: np.expand_dims(image,0), self.annotations: np.expand_dims(annotation,0), self.is_training_placeholder: False}
		feat, pred, loss = sess.run([self.vgg_output_flat, self.pred, self.cross_entropy_sum], feed_dict=feed_dict)
		source_features[n] = feat
		source_preds[n] = pred
		source_losses[n] = loss
	    
	    print 'Average source loss: ' + str(source_losses.mean())
	    
	    print 'Evaluating TARGET - ' + seq_2_name
	    
	    for n, image, annotation in zip(range(len(target_images)), target_images, target_annotations):
		
		if n%100==0:
		    print n 
		feed_dict = {self.images: np.expand_dims(image,0), self.annotations: np.expand_dims(annotation,0), self.is_training_placeholder: False}
		feat, pred, loss = sess.run([self.vgg_output_flat, self.pred, self.cross_entropy_sum], feed_dict=feed_dict)
		target_features[n] = feat
		target_preds[n] = pred
		target_losses[n] = loss
		
	    print 'Average target loss: ' + str(target_losses.mean())
	    print 'break'
	    
    def extract_VGG16_features(self, source_images):
	
	print 'Extracting VGG_16 features.'
	
	self.build_model(mode='train_semantic_extractor')
	
	source_features = np.zeros((len(source_images),128))

	with tf.Session() as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    self.read_vgg_weights_except_fc8_func(sess)
	    sess.run(self.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, './experiments/'+self.seq_name+'/model/segm_model')
	    
	    print 'Extracting VGG-16 features from ' + self.seq_name
	    
	    for n, image in enumerate(source_images):
		
		if n%100==0:
		    print n 
		    
		feed_dict = {self.images: np.expand_dims(image,0), self.annotations: np.zeros((1,224,224,1)), self.is_training_placeholder: False}
		feat = sess.run(self.vgg_output_flat, feed_dict=feed_dict)
		source_features[n] = feat
	    
	    return source_features
	    
    def train_feature_generator(self):
	
	epochs=10000
	batch_size=32
	noise_dim=100

	summary_string_writer = tf.summary.FileWriter(self.log_dir)

	config = tf.ConfigProto(device_count = {'GPU': 0})

	source_images, source_annotations = load_synthia(self.seq_name, no_elements=900)
		
	source_features = self.extract_VGG16_features(source_images)
	
	self.build_model(mode='train_feature_generator')
	
        with tf.Session() as sess:
	
	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()
	    
            tf.global_variables_initializer().run()
	    
	    t = 0
	    
	    for i in range(epochs):
		
		#~ print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(source_images), batch_size), range(batch_size, len(source_images), batch_size)):
		    
		    t += 1

		    Z_samples = utils.sample_Z(batch_size, noise_dim, 'uniform')

		    feed_dict = {self.noise: Z_samples, self.fx: source_features[start:end]}
	    
		    avg_D_fake = sess.run(self.logits_fake, feed_dict)
		    avg_D_real = sess.run(self.logits_real, feed_dict)
		    
		    sess.run(self.d_train_op, feed_dict)
		    sess.run(self.g_train_op, feed_dict)
		    
		    if (t+1) % 500 == 0:
			summary, dl, gl = sess.run([self.summary_op, self.d_loss, self.g_loss], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
				   %(t+1, int(epochs*len(source_images) /batch_size), dl, gl))
			print 'avg_D_fake',str(avg_D_fake.mean()),'avg_D_real',str(avg_D_real.mean())
			
                    if (t+1) % 5000 == 0:  
			saver.save(sess, './experiments/'+self.seq_name+'/model/sampler')
 
    def plot_tsne(self, seq_2_name = '...'):
	
	source_images, _ = load_synthia(self.seq_name, no_elements=900)
	target_images, _ = load_synthia(seq_2_name, no_elements=900)
	
	source_features = self.extract_VGG16_features(source_images)
	tf.reset_default_graph()
	target_features = self.extract_VGG16_features(target_images)
	tf.reset_default_graph()
	
	self.build_model(mode='train_feature_generator')
			
	self.config = tf.ConfigProto(device_count = {'GPU': 0})

        with tf.Session() as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
	    
	    print ('Loading feature generator.')
	    variables_to_restore = slim.get_model_variables(scope='feature_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, './experiments/'+self.seq_name+'/model/sampler')
	    
	    n_samples = 900
            noise = utils.sample_Z(n_samples,100,'uniform')
	    
	    feed_dict = {model.noise: noise, model.fx: source_features[1:2]}
	    
	    fzy = sess.run([model.fzy], feed_dict)
	    
	    with open('./experiments/'+self.seq_name+'/features.pkl','w') as f:
		cPickle.dump((source_features, target_features, fzy), f, cPickle.HIGHEST_PROTOCOL)

	    #~ print 'Computing T-SNE.'
#~ 
	    #~ model = TSNE(n_components=2, random_state=0)
#~ 
	       #~ 
	    #~ if sys.argv[1] == '1':
		#~ TSNE_hA = model.fit_transform(source_features)
		#~ 
		#~ f, ax = plt.plot()
		#~ ax.set_facecolor('white')
		#~ 
		#~ ax.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.ones((n_samples)), s=3, cmap = mpl.cm.jet)
		#~ 
	    #~ elif sys.argv[1] == '2':
		#~ TSNE_hA = model.fit_transform(np.vstack((fzy,source_features)))
		#~ 
		#~ f, ax = plt.plot()
		#~ ax.set_facecolor('white')
		#~ 
		#~ ax.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)))), s=3, cmap = mpl.cm.jet, alpha=0.5)
#~ 
#~ 
	    #plt.legend()
	    #~ plt.show()
	    
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################


if __name__ == "__main__":

    model = DSN(seq_name='SYNTHIA-SEQS-01-FALL')

    #~ print 'Training feature generator'
    #~ 
    #~ model.plot_tsne(seq_2_name = 'SYNTHIA-SEQS-01-NIGHT')
    
    #~ print 'Evaluating model.'
 
    #~ model.eval_semantic_extractor(seq_2_name='SYNTHIA-SEQS-01-SPRING')
 
    print 'Training semantic extractor'
    
    model.train_semantic_extractor()
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    

#TO PLOT IMAGES





#~ cmap = plt.get_cmap('bwr')

#~ f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#~ ax1.imshow(np.uint8(pred_np.squeeze() != 1), vmax=1.5, vmin=-0.4, cmap=cmap)
#~ ax1.set_title('Argmax. Iteration # ' + str(i))
#~ probability_graph = ax2.imshow(probabilities_np.squeeze()[:, :, 0])
#~ ax2.set_title('Probability of the Class. Iteration # ' + str(i))

#~ plt.colorbar(probability_graph)
#~ plt.show()






#~ f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#~ 
#~ cmap = plt.get_cmap('bwr')
#~ 
#~ ax1.imshow(np.uint8(final_predictions.squeeze() != 1),
	   #~ vmax=1.5,
	   #~ vmin=-0.4,
	   #~ cmap=cmap)
#~ 
#~ ax1.set_title('Final Argmax')
#~ 
#~ probability_graph = ax2.imshow(final_probabilities.squeeze()[:, :, 0])
#~ ax2.set_title('Final Probability of the Class')
#~ plt.colorbar(probability_graph)
#~ 
#~ plt.show()
