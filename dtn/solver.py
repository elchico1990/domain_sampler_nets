import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
import pickle
import os
import scipy.io
import scipy.misc
import cPickle
import sys
import glob
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import utils
from sklearn.manifold import TSNE

from scipy import misc


class Solver(object):

    def __init__(self, model, batch_size=64, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 svhn_dir='svhn', syn_dir='syn', mnist_dir='mnist', mnist_m_dir='mnist_m', usps_dir='usps', log_dir='logs', sample_save_path='sample', 
                 model_save_path='model', pretrained_model='model/model', pretrained_sampler='model/sampler', 
		 test_model='model/dtn', convdeconv_model = 'model/conv_deconv'):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.svhn_dir = svhn_dir
        self.syn_dir = syn_dir
        self.mnist_dir = mnist_dir
        self.mnist_m_dir = mnist_dir
        self.usps_dir = usps_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
	self.pretrained_sampler = pretrained_sampler
        self.test_model = test_model
	self.convdeconv_model = convdeconv_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	self.protocol = 'syn_svhn'

    def load_svhn(self, image_dir, split='train'):
        print ('Loading SVHN dataset.')
        
        image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        return images, labels

    def load_syn(self, image_dir, split='train'):
        print ('Loading SYN dataset.')
        
        image_file = 'synth_train_32x32.mat' if split=='train' else 'synth_test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        syn = scipy.io.loadmat(image_dir)
        images = np.transpose(syn['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = syn['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        return images, labels

    def load_mnist(self, image_dir, split='train'):
        print ('Loading MNIST dataset.')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        return images, labels

    def load_mnist_m(self,image_dir, split='train'):
	
	if split == 'train':
	    data_dir = image_dir + '/mnist_m_train/'
	    with open(image_dir + '/mnist_m_train_labels.txt') as f:
		content = f.readlines()
	elif split == 'test':
	    data_dir = image_dir + '/mnist_m_test/'
	    with open(image_dir + '/mnist_m_test_labels.txt') as f:
		content = f.readlines()
	
	
	content = [c.split('\n') for c in content]
	images_files = [c.split(' ')[0] for c in content]
	labels = np.array([int(c.split(' ')[1]) for c in content]).reshape(-1)
	
	images = np.zeros(len(labels), 32, 32, 3)
	
	for no_img,img in enumerate(images_files):
	    img_dir = data_dir + img
	    im = misc.imread(img_dir)
	    im = np.expand_dims(im, axis=0)
	    images[no_img] = im
	
	return images, labels

    def load_usps(self, image_dir):
        
	print ('Loading USPS dataset.')
        image_file = 'train.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            usps = pickle.load(f)
        images = usps['X'] / 127.5 - 1
        labels = usps['y']
	labels -= 1
	labels[labels==255] = 0
	random_idx = np.arange(len(labels))
	npr.seed(123)
	npr.shuffle(random_idx)
	images = images[random_idx]
	labels = labels[random_idx]
        return images, np.squeeze(labels)

    def load_gen_images(self):
	
	'''
	Loading images generated with eval_dsn()
	Assuming that image_dir contains folder with
	subfolders 1,2,...,9.
	'''
	
	print 'Loading generated images.'
	
	no_images = 1000 # number of images per digit
	
	labels = np.zeros((10 * no_images,)).astype(int)
	images = np.zeros((10 * no_images,28,28,1))
	
	for l in range(10):
	    print l
	    counter = 0
	    for img_dir in sorted(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/*'))[:no_images]:
		im = misc.imread(img_dir, mode='L')
		im = np.expand_dims(im, axis=0)
		im = np.expand_dims(im, axis=3)
		images[l * no_images + counter] = im
		labels[l * no_images + counter] = l
		counter+=1
	    
	print 'break'
	images = images / 127.5 - 1
	return images, labels
	
    def pretrain(self):
	
	print 'Pretraining.'
        
	if self.protocol == 'svhn_mnist':	
	    
	    src_images, src_labels = self.load_svhn(self.svhn_dir, split='train')
	    src_test_images, src_test_labels = self.load_svhn(self.svhn_dir, split='test')
	    
	    trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train')
	    trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')
        
	elif self.protocol == 'syn_svhn':	
	    
	    src_images, src_labels = self.load_syn(self.syn_dir, split='train')
	    src_test_images, src_test_labels = self.load_syn(self.syn_dir, split='test')
	    
	    trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='train')
	    trg_test_images, trg_test_labels = self.load_svhn(self.svhn_dir, split='test')

	elif self.protocol == 'mnist_usps':	
	    
	    src_images, src_labels = self.load_mnist(self.mnist_dir, split='train')
	    src_images = src_images[:2000]
	    src_labels = src_labels[:2000]
	    src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')
	    
	    trg_images, trg_labels = self.load_usps(self.usps_dir)
	    trg_images = trg_images[:1800]
	    trg_labels = trg_labels[:1800]
	    trg_test_images = trg_images
	    trg_test_labels = trg_labels
	    
	#~ print trg_labels[1]
	#~ print src_labels[1]
	#~ plt.figure(1)
	#~ plt.imshow(np.squeeze(trg_images[1]))
	#~ plt.figure(2)
	#~ plt.imshow(np.squeeze(src_images[1]))
	#~ plt.show()

        # build a graph
        model = self.model
        model.build_model()
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            print ('Loading pretrained model.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    epochs = 100
	    
	    t = 0

	    for i in range(epochs):
		
		print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(src_images), self.batch_size), range(self.batch_size, len(src_images), self.batch_size)):
		    
		    t+=1
		       
		    feed_dict = {model.src_images: src_images[start:end], model.src_labels: src_labels[start:end], model.trg_images: trg_images[0:2], model.trg_labels: trg_labels[0:2]} #trg here is just needed by the model but otherwise useless. 
		    
		    sess.run(model.train_op, feed_dict) 

		    if (t+1) % 250 == 0:
			summary, l, src_acc = sess.run([model.summary_op, model.loss, model.src_accuracy], feed_dict)
			src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:1000]
			trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:3000]
			test_src_acc, test_trg_acc, _ = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.loss], 
					       feed_dict={model.src_images: src_test_images[src_rand_idxs], 
							  model.src_labels: src_test_labels[src_rand_idxs],
							  model.trg_images: trg_test_images[trg_rand_idxs], 
							  model.trg_labels: trg_test_labels[trg_rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] src test acc [%.2f] trg test acc [%.4f]' \
				   %(t+1, self.pretrain_iter, l, src_acc, test_src_acc, test_trg_acc))
			
		    if (t+1) % 250 == 0:
			#~ print 'Saved.'
			saver.save(sess, os.path.join(self.model_save_path, 'model'))

    def train_convdeconv(self):

        trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train')
        trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    epochs = 100
	    
	    t = 0

	    for i in range(epochs):
		
		print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(trg_images), self.batch_size), range(self.batch_size, len(trg_images), self.batch_size)):
		    
		    t+=1
		       
		    feed_dict = {model.images: trg_images[start:end]}
		    
		    sess.run(model.train_op, feed_dict) 

		    if (t+1) % 250 == 0:
			rand_idxs = np.random.permutation(trg_test_images.shape[0])[:1000]
			summary, l = sess.run([model.summary_op, model.loss], feed_dict = {model.images: trg_test_images[rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f]' \
				   %(t+1, self.pretrain_iter, l))
			
		    if (t+1) % 250 == 0:
			#~ print 'Saved.'
			saver.save(sess, os.path.join(self.model_save_path, 'conv_deconv'))
	    
    def train_sampler(self):
	
	print 'Training sampler.'
        
	if self.protocol == 'svhn_mnist':	
	    source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	    source_labels = utils.one_hot(source_labels, 10)
        
	elif self.protocol == 'syn_svhn':	
	    source_images, source_labels = self.load_syn(self.syn_dir, split='train')
	    source_labels = utils.one_hot(source_labels, 10)

	elif self.protocol == 'mnist_usps':	
	    source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	    source_labels = utils.one_hot(source_labels, 10)
	    source_images = source_images[:2000]
	    source_labels = source_labels[:2000]
		
	
	
	
	#~ svhn_images = svhn_images[np.where(np.argmax(svhn_labels,1)==1)]
	#~ svhn_labels = svhn_labels[np.where(np.argmax(svhn_labels,1)==1)]
        
        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	
	batch_size = self.batch_size
	noise_dim = 100
	epochs = 5000

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
            print ('Loading pretrained model.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            # restore variables of F
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()
	    
	    #~ feed_dict = {model.images: source_images[:10000]}
	    #~ fx = sess.run(model.fx, feed_dict)
		 	    
	    t = 0
	    
	    for i in range(epochs):
		
		#~ print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(source_images), batch_size), range(batch_size, len(source_images), batch_size)):
		    
		    t += 1

		    Z_samples = utils.sample_Z(batch_size, noise_dim, 'uniform')

		    feed_dict = {model.noise: Z_samples, model.images: source_images[start:end], model.labels: source_labels[start:end]}
	    
		    avg_D_fake = sess.run(model.logits_fake, feed_dict)
		    avg_D_real = sess.run(model.logits_real, feed_dict)
		    
		    sess.run(model.d_train_op, feed_dict)
		    sess.run(model.g_train_op, feed_dict)
		    
		    if (t+1) % 100 == 0:
			summary, dl, gl = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
				   %(t+1, int(epochs*len(source_images) /batch_size), dl, gl))
			print 'avg_D_fake',str(avg_D_fake.mean()),'avg_D_real',str(avg_D_real.mean())
			
                    if (t+1) % 1000 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'sampler')) 

    def train_dsn(self):
        
	print 'Training DSN.'
	
	if self.protocol=='svhn_mnist':
	    source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	    target_images, target_labels = self.load_mnist(self.mnist_dir, split='train')

	if self.protocol=='syn_svhn':
	    source_images, source_labels = self.load_syn(self.syn_dir, split='train')
	    target_images, target_labels = self.load_svhn(self.svhn_dir, split='train')

	elif self.protocol=='mnist_usps':
	    source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	    target_images, target_labels = self.load_usps(self.usps_dir)
	    source_images = source_images[:2000]
	    source_labels = source_labels[:2000]
	    target_images = target_images[:1800]
	    target_labels = target_labels[:1800]
	
        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

	with tf.Session(config=self.config) as sess:
	    	    
	    # initialize G and D
	    tf.global_variables_initializer().run()
	    # restore variables of F
	    
	    print ('Loading pretrained encoder.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_model)
	    
	    #~ print ('Loading pretrained encoder disc.')
	    #~ variables_to_restore = slim.get_model_variables(scope='disc_e')
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, self.pretrained_sampler)
	    
	    #~ print ('Loading pretrained G.')
	    #~ variables_to_restore = slim.get_model_variables(scope='generator')
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, self.test_model)
	    
	    #~ print ('Loading pretrained D_g.')
	    #~ variables_to_restore = slim.get_model_variables(scope='disc_g')
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, self.test_model)
	    
	    print ('Loading sample generator.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    

	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
	    saver = tf.train.Saver()

	    print ('Start training.')
	    trg_count = 0
	    t = 0
	    
	    self.batch_size = 128
	    
	    label_gen = utils.one_hot(np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,9,9,9]),10)
	    
	    for step in range(10000000):
		
		trg_count += 1
		t+=1
		
		i = step % int(source_images.shape[0] / self.batch_size)
		j = step % int(target_images.shape[0] / self.batch_size)
		
		src_images = source_images[i*self.batch_size:(i+1)*self.batch_size]
		src_labels = utils.one_hot(source_labels[i*self.batch_size:(i+1)*self.batch_size],10)
		src_labels_int = source_labels[i*self.batch_size:(i+1)*self.batch_size]
		src_noise = utils.sample_Z(self.batch_size,100,'uniform')
		trg_images = target_images[j*self.batch_size:(j+1)*self.batch_size]
		
		feed_dict = {model.src_images: src_images, model.src_noise: src_noise, model.src_labels: src_labels, model.trg_images: trg_images, model.labels_gen: label_gen}
		
		sess.run(model.E_train_op, feed_dict) 
		sess.run(model.DE_train_op, feed_dict)
		
		#~ sess.run(model.const_train_op, feed_dict)		
		#~ sess.run(model.G_train_op, feed_dict) 
		#~ sess.run(model.DG_train_op, feed_dict) 

		logits_E_real,logits_E_fake = sess.run([model.logits_E_real,model.logits_E_fake],feed_dict) 
		
		if (step+1) % 10 == 0:
		    
		    summary, E, DE = sess.run([model.summary_op, model.E_loss, model.DE_loss], feed_dict)
		    summary_writer.add_summary(summary, step)
		    print ('Step: [%d/%d] E: [%.6f] DE: [%.6f] E_real: [%.2f] E_fake: [%.2f]' \
			       %(step+1, self.train_iter, E, DE, logits_E_real.mean(),logits_E_fake.mean()))

		    

		if (step+1) % 100 == 0:
		    saver.save(sess, os.path.join(self.model_save_path, 'dtn'))
	
    def eval_dsn(self):
        # build model
        model = self.model
        model.build_model()

	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
        with tf.Session(config=self.config) as sess:
	    
	    print ('Loading pretrained G.')
	    variables_to_restore = slim.get_model_variables(scope='generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.test_model)
	    
	    
	    print ('Loading sample generator.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    
	    for n in range(10):
	    
		if self.protocol=='svhn_mnist':
		    source_images, source_labels = self.load_svhn(self.svhn_dir)
		    source_labels[:] = str(n)
		elif self.protocol=='mnist_usps':
		    source_images, source_labels = self.load_mnist(self.mnist_dir)
		    source_labels[:] = str(n)

		# train model for source domain S
		src_labels = utils.one_hot(source_labels[:10000],10)
		src_noise = utils.sample_Z(10000,100,'uniform')

		feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels}

		samples = sess.run(model.sampled_images, feed_dict)

		for i in range(1000):
		    
		    print str(i)+'/'+str(len(samples)), np.argmax(src_labels[i])
		    plt.imshow(np.squeeze(samples[i]), cmap='gray')
		    plt.imsave('./sample/'+str(np.argmax(src_labels[i]))+'/'+str(i)+'_'+str(np.argmax(src_labels[i])),np.squeeze(samples[i]), cmap='gray')

    def train_gen_images(self):
        # load svhn dataset
        src_images, src_labels = self.load_gen_images()
	
	random_idx = np.arange(len(src_images))
	npr.shuffle(random_idx)
	src_images = src_images[random_idx]
	src_labels = src_labels[random_idx]
	
	if self.protocol == 'svhn_mnist':
	    trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='test')
	if self.protocol == 'syn_svhn':
	    trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='test')
	elif self.protocol == 'mnist_usps':
	    trg_images, trg_labels = self.load_usps(self.usps_dir)
	    trg_images = trg_images[:1800]
	    trg_labels = trg_labels[:1800]
	
        # build a graph
        model = self.model
        model.build_model()
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            #~ print ('Loading pretrained model.')
            #~ variables_to_restore = slim.get_model_variables(scope='encoder')
            #~ restorer = tf.train.Saver(variables_to_restore)
            #~ restorer.restore(sess, self.test_model)
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
	    
	    epochs = 100
	    
	    t = 0
	    
	    for i in range(epochs):
		
		print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(src_images), self.batch_size), range(self.batch_size, len(src_images), self.batch_size)):
		    
		    t+=1
		       
		    feed_dict = {model.src_images: src_images[start:end], model.src_labels: src_labels[start:end], model.trg_images: trg_images[0:2], model.trg_labels: trg_labels[0:2]} #trg here is just needed by the model but otherwise useless. 
		    
		    sess.run(model.train_op, feed_dict) 

		    if (t+1) % 250 == 0:
			summary, l, src_acc = sess.run([model.summary_op, model.loss, model.src_accuracy], feed_dict)
			src_rand_idxs = np.random.permutation(src_images.shape[0])[:1000]
			trg_rand_idxs = np.random.permutation(trg_images.shape[0])[:]
			test_acc = sess.run(model.trg_accuracy, 
					       feed_dict={model.src_images: src_images[src_rand_idxs], 
							  model.src_labels: src_labels[src_rand_idxs],
							  model.trg_images: trg_images[trg_rand_idxs], 
							  model.trg_labels: trg_labels[trg_rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f] train acc: [%.3f] test acc [%.3f]' \
				   %(t+1, self.pretrain_iter, l, src_acc, test_acc))
			
		    if (t+1) % 250 == 0:
			#~ print 'Saved.'
			saver.save(sess, os.path.join(self.model_save_path, 'model_gen'))

    def check_TSNE(self):
	
	if self.protocol == 'svhn_mnist':
	    source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	    target_images, target_labels = self.load_mnist(self.mnist_dir, split='train')
	if self.protocol == 'syn_svhn':
	    source_images, source_labels = self.load_syn(self.syn_dir, split='train')
	    target_images, target_labels = self.load_svhn(self.svhn_dir, split='train')
	elif self.protocol == 'mnist_usps':
	    source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	    target_images, target_labels = self.load_usps(self.usps_dir)
	    source_images = source_images[:2000]
	    source_labels = source_labels[:2000]
	    target_images = target_images[:1800]
	    target_labels = target_labels[:1800]
	

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
			
	self.config = tf.ConfigProto(device_count = {'GPU': 0})

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
	    
	    if sys.argv[1] == 'test':
		print ('Loading test model.')
		variables_to_restore = slim.get_model_variables(scope='encoder')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.test_model)
	    
	    elif sys.argv[1] == 'pretrain':
		print ('Loading pretrained model.')
		variables_to_restore = slim.get_model_variables(scope='encoder')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.pretrained_model)
		
	    elif sys.argv[1] == 'convdeconv':
		print ('Loading convdeconv model.')
		variables_to_restore = slim.get_model_variables(scope='conv_deconv')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.convdeconv_model)
		
	    else:
		raise NameError('Unrecognized mode.')
	    
            
	    n_samples = 500
            src_labels = utils.one_hot(source_labels[:n_samples],10)
	    trg_labels = utils.one_hot(target_labels[:n_samples],10)
	    src_noise = utils.sample_Z(n_samples,100,'uniform')
	   
	    
	    if sys.argv[1] == 'convdeconv':
	    
		feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.src_images: source_images, model.trg_images: target_images[:n_samples]}
		h_repr = sess.run(model.h_repr, feed_dict)
		
	    else:
	    
		print ('Loading sampler.')
		variables_to_restore = slim.get_model_variables(scope='sampler_generator')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.pretrained_sampler)
	
		
		feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.src_images: source_images[:n_samples], model.trg_images: target_images[:n_samples]}
		
		fzy, fx_src, fx_trg = sess.run([model.fzy, model.fx_src, model.fx_trg], feed_dict)
		
		src_labels = np.argmax(src_labels,1)
		trg_labels = np.argmax(trg_labels,1)

	    print 'Computing T-SNE.'

	    model = TSNE(n_components=2, random_state=0)

	       
	    if sys.argv[2] == '1':
		TSNE_hA = model.fit_transform(np.vstack((fx_src)))
		plt.figure(2)
		plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples)))), s=3, cmap = mpl.cm.jet)
		plt.figure(3)
		plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels)), s=3,  cmap = mpl.cm.jet)
		
	    elif sys.argv[2] == '2':
		TSNE_hA = model.fit_transform(np.vstack((fzy,fx_src)))
	        plt.figure(2)
		plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)))), s=3, cmap = mpl.cm.jet)
		plt.figure(3)
                plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels,src_labels)), s=3, cmap = mpl.cm.jet)

	    elif sys.argv[2] == '3':
		TSNE_hA = model.fit_transform(np.vstack((fzy,fx_src,fx_trg)))
	        plt.figure(2)
		plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels, src_labels, trg_labels, )), s=5,  cmap = mpl.cm.jet)
		plt.figure(3)
                plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)), 3 * np.ones((n_samples,)))), s=5,  cmap = mpl.cm.jet)

	    elif sys.argv[2] == '4':
		TSNE_hA = model.fit_transform(h_repr)
	        plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.argmax(trg_labels,1), s=3,  cmap = mpl.cm.jet)
		

	    plt.legend()
	    plt.show()
	    
    def test(self):
	
	if self.protocol == 'svhn_mnist':
	    
	    src_images, src_labels = self.load_svhn(self.svhn_dir, split='train')
	    src_test_images, src_test_labels = self.load_svhn(self.svhn_dir, split='test')
	    
	    trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train')
	    trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')
	
	elif self.protocol == 'syn_svhn':
	    
	    src_images, src_labels = self.load_syn(self.syn_dir, split='train')
	    src_test_images, src_test_labels = self.load_syn(self.syn_dir, split='test')
	    
	    trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='test')
	    trg_test_images, trg_test_labels = self.load_svhn(self.svhn_dir, split='test')
	
	if self.protocol == 'mnist_usps':
	    
	    src_images, src_labels = self.load_mnist(self.mnist_dir, split='train')
	    src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')
	    src_images = src_images[:2000]
	    src_labels = src_labels[:2000]
	    
	    trg_images, trg_labels = self.load_usps(self.usps_dir)
	    trg_images = trg_images[:1800]
	    trg_labels = trg_labels[:1800]
	    trg_test_images = trg_images
	    trg_test_labels = trg_labels
	    
	#~ gen_images, gen_labels = self.load_gen_images()

	# build a graph
	model = self.model
	model.build_model()
	
	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
	with tf.Session(config=self.config) as sess:
	    tf.global_variables_initializer().run()
	    saver = tf.train.Saver()
	    
	    t = 0
	    
	    acc = []
	    
	    while(True):
		
		if sys.argv[1] == 'test':
		    print ('Loading test model.')
		    variables_to_restore = slim.get_model_variables(scope='encoder')
		    restorer = tf.train.Saver(variables_to_restore)
		    restorer.restore(sess, self.test_model)
		
		elif sys.argv[1] == 'pretrain':
		    print ('Loading pretrained model.')
		    variables_to_restore = slim.get_model_variables(scope='encoder')
		    restorer = tf.train.Saver(variables_to_restore)
		    restorer.restore(sess, self.pretrained_model)
		    
		else:
		    raise NameError('Unrecognized mode.')
	    
		t+=1
    
		src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:]
		trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:]
		test_src_acc, test_trg_acc, _ = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.loss], 
				       feed_dict={model.src_images: src_test_images[src_rand_idxs], 
						  model.src_labels: src_test_labels[src_rand_idxs],
						  model.trg_images: trg_test_images[trg_rand_idxs], 
						  model.trg_labels: trg_test_labels[trg_rand_idxs]})
		src_acc = sess.run(model.src_accuracy, feed_dict={model.src_images: src_images[:20000], 
								  model.src_labels: src_labels[:20000],
						                  model.trg_images: trg_test_images[trg_rand_idxs], 
								  model.trg_labels: trg_test_labels[trg_rand_idxs]})
						  
		print ('Step: [%d/%d] src train acc [%.3f]  src test acc [%.3f] trg test acc [%.3f]' \
			   %(t+1, self.pretrain_iter, src_acc, test_src_acc, test_trg_acc))
			   
		acc.append(test_trg_acc)
		with open('test_acc.pkl', 'wb') as f:
		    cPickle.dump(acc,f,cPickle.HIGHEST_PROTOCOL)
    
		#~ gen_acc = sess.run(fetches=[model.trg_accuracy, model.trg_pred], 
				       #~ feed_dict={model.src_images: gen_images, 
						  #~ model.src_labels: gen_labels,
						  #~ model.trg_images: gen_images, 
						  #~ model.trg_labels: gen_labels})
				  
		#~ print ('Step: [%d/%d] src train acc [%.2f]  src test acc [%.2f] trg test acc [%.2f]' \
			   #~ %(t+1, self.pretrain_iter, gen_acc))
	
		time.sleep(.5)
		    
if __name__=='__main__':

    from model import DSN
    model = DSN(mode='eval_dsn', learning_rate=0.0003)
    solver = Solver(model)
    solver.check_TSNE()
