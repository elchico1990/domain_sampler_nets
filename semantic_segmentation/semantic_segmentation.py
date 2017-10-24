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
from utils import *

class DSN(object):
    
    def __init__(self, seq_name, no_classes=13):

	self.seq_name = seq_name
	self.no_classes = no_classes
	self.log_folder = './logs'
	self.vgg_checkpoint_path = './vgg_16.ckpt'
	
    def vgg_encoding(self, processed_images, is_training_placeholder): 
		
	with slim.arg_scope(vgg.vgg_arg_scope()):

	    fc7 = vgg.vgg_16(processed_images,
				num_classes=self.no_classes,
				is_training=is_training_placeholder,
				spatial_squeeze=False,
				fc_conv_padding='VALID',
				reuse=False,
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
	    
    def build_model(self):
	
	self.image_tensor = tf.placeholder(tf.float32, [None, 224 * 1,224 * 1, 3], 'images')
	self.annotation_tensor = tf.placeholder(tf.float32, [None, 224 * 1,224 * 1, 1], 'annotations')
	self.is_training_placeholder = tf.placeholder(tf.bool)

	labels_tensors = [tf.to_float(tf.equal(self.annotation_tensor, i)) for i in range(self.no_classes)]

	try:
	    combined_mask = tf.concat(axis=3, values = labels_tensors)
	except:
	    combined_mask = tf.concat(3,labels_tensors)
	    
	flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, self.no_classes))

	image_float = tf.to_float(self.image_tensor, name='ToFloat')

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

    def train_model(self):

	summary_string_writer = tf.summary.FileWriter(model.log_folder)

	config = tf.ConfigProto(device_count = {'GPU': 0})

	with tf.Session() as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    model.read_vgg_weights_except_fc8_func(sess)
	    sess.run(model.vgg_fc8_weights_initializer)
     
	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    model.read_vgg_weights_except_fc8_func(sess)
	    sess.run(model.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, './tensorflow_models/'+self.seq_name+'/segm_model')
	    
	    saver = tf.train.Saver(model.train_vars)
	    
	    with open('./tensorflow_models/'+self.seq_name+'/train_vars.pkl','w') as f:
		cPickle.dump(model.train_vars,f,cPickle.HIGHEST_PROTOCOL)
		
	    print 'break'

	    images, annotations = load_synthia(self.seq_name, no_elements=900)

	    feed_dict = {model.image_tensor: images,
			 model.annotation_tensor: annotations,
			 model.is_training_placeholder: False}

	    EPOCHS = 1000
	    BATCH_SIZE = 1

	    for e in range(EPOCHS):
		
		losses = []
		
		print e
		
		for n, start, end in zip(range(len(images)), range(0,len(images),BATCH_SIZE), range(BATCH_SIZE,len(images),BATCH_SIZE)):
			    
		    feed_dict = {model.image_tensor: images[start:end], model.annotation_tensor: annotations[start:end], model.is_training_placeholder: True}

		    loss, summary_string = sess.run([model.cross_entropy_sum, model.merged_summary_op], feed_dict=feed_dict)

		    sess.run(model.train_op, feed_dict=feed_dict)

		    summary_string_writer.add_summary(summary_string, e)

		    
		    if n%10==0:
			print e,'-',n
			losses.append(loss)
			print("Current Average Loss: " + str(np.array(losses).mean()))
		pred_np, probabilities_np = sess.run([model.pred, model.probabilities], feed_dict={model.image_tensor: images[1:2], model.annotation_tensor: annotations[1:2], model.is_training_placeholder: False})
		plt.imsave('./images/'+str(e)+self.seq_name+'/'+'.png', np.squeeze(pred_np))	    
		saver.save(sess, './tensorflow_models/'+self.seq_name+'/segm_model')

	    feed_dict[model.is_training_placeholder] = False
	    feed_dict = {model.feature_tensor: vgg_features[1:2],model.annotation_tensor: annotations[1:2],model.is_training_placeholder: False}


	    
	    pred, probabilities, labels_tensors, combined_mask, logits, upsampled_logits, flat_logits, processed_images, train_images, train_annotations = sess.run([pred, probabilities, labels_tensors, combined_mask, logits, upsampled_logits, flat_logits, processed_images, image_tensor, annotation_tensor],
						     feed_dict=feed_dict)
				

	    final_predictions, final_probabilities, final_loss = sess.run([pred,
									   probabilities,
									   cross_entropy_sum],
									  feed_dict=feed_dict)



	    print("Final Loss: " + str(final_loss))

	    summary_string_writer.close()

    def eval_model(self, seq_2_name):

	summary_string_writer = tf.summary.FileWriter(model.log_folder)

	config = tf.ConfigProto(device_count = {'GPU': 0})

	with tf.Session() as sess:
		
	    print 'Loading weights.'

	    #~ # Run the initializers.
	    sess.run(tf.global_variables_initializer())
	    model.read_vgg_weights_except_fc8_func(sess)
	    sess.run(model.vgg_fc8_weights_initializer)
	    variables_to_restore = [i for i in slim.get_model_variables() if ('fc7' in i.name) or ('semantic_extractor' in i.name)]
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, './tensorflow_models/'+self.seq_name+'/segm_model')
	    
	    saver = tf.train.Saver(model.train_vars)

	    source_images, source_annotations = load_synthia(self.seq_name, no_elements=900)
	    target_images, target_annotations = load_synthia(seq_2_name, no_elements=900)
			 
	    source_features = np.zeros((len(source_images),128))
	    target_features = np.zeros((len(target_images),128))
	    source_losses = np.zeros((len(source_images), 1))
	    
	    source_preds = np.zeros((len(source_images),224,224))
	    target_preds = np.zeros((len(target_images),224,224))
	    target_losses = np.zeros((len(target_images), 1))
	    
	    print 'Evaluating SOURCE - ' + self.seq_name
	    
	    for n, image, annotation in zip(range(len(source_images)), source_images, source_annotations):
		
		if n%100==0:
		    print n 
		feed_dict = {model.image_tensor: np.expand_dims(image,0), model.annotation_tensor: np.expand_dims(annotation,0), model.is_training_placeholder: False}
		feat, pred, loss = sess.run([model.vgg_output_flat, model.pred, model.cross_entropy_sum], feed_dict=feed_dict)
		source_features[n] = feat
		source_preds[n] = pred
		source_losses[n] = loss
	    
	    print 'Average source loss: ' + str(source_losses.mean())
	    
	    print 'Evaluating TARGET - ' + seq_2_name
	    
	    for n, image, annotation in zip(range(len(target_images)), target_images, target_annotations):
		
		if n%100==0:
		    print n 
		feed_dict = {model.image_tensor: np.expand_dims(image,0), model.annotation_tensor: np.expand_dims(annotation,0), model.is_training_placeholder: False}
		feat, pred, loss = sess.run([model.vgg_output_flat, model.pred, model.cross_entropy_sum], feed_dict=feed_dict)
		target_features[n] = feat
		target_preds[n] = pred
		target_losses[n] = loss
		
	    print 'Average target loss: ' + str(target_losses.mean())
	    print 'break'



if __name__ == "__main__":

    model = DSN(seq_name='SYNTHIA-SEQS-01-DAWN')
    
    print 'Building model.'
    
    model.build_model()
    
    print 'Evaluating model.'
    
    model.eval_model(seq_2_name='SYNTHIA-SEQS-01-SPRING')

    
    
    #~ model.train_model()
    

	
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    

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
