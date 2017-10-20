import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
import pickle
import os
import scipy.io
import scipy.misc
from scipy.misc import imsave
import cPickle
import sys
import glob
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import utils
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from scipy import misc

class Solver(object):

    def __init__(self, model, batch_size=16, pretrain_iter=100000, train_iter=51000, sample_iter=2000, 
                 svhn_dir='../data/svhn', syn_dir='../data/syn', mnist_dir='mnist', mnist_m_dir='mnist_m', usps_dir='usps', amazon_dir='amazon_reviews',
		 log_dir='logs', sample_save_path='sample', model_save_path='model', pretrained_model='model/model', gen_model='model/model_gen', pretrained_sampler='model/sampler', 
		 test_model='model/dtn', convdeconv_model = 'model/conv_deconv', start_img=0, end_img=1600):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.svhn_dir = svhn_dir
        self.syn_dir = syn_dir
        self.mnist_dir = 'data/'+mnist_dir
        self.mnist_m_dir = 'data/'+mnist_m_dir
        self.usps_dir = 'data/'+usps_dir
	self.amazon_dir = 'data/'+amazon_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
	self.gen_model = gen_model
	self.pretrained_sampler = pretrained_sampler
        self.test_model = test_model
	self.convdeconv_model = convdeconv_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	self.protocol = 'syn_svhn' # possibilities: svhn_mnist, mnist_usps, usps_mnist, syn_svhn, mnist_mnist_m, amazon_reviews
	
	self.start_img = start_img
	self.end_img = end_img
    
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

    def load_gen_images_NOT_USED(self):
	
	'''
	Loading images generated with eval_dsn()
	Assuming that ./sample contains folder with
	subfolders 1,2,...,9.
	'''
	
	print 'Loading generated images.'
	
	no_images = 3000 # number of images per digit
	
	labels = np.zeros((10 * no_images,)).astype(int)
	images = np.zeros((10 * no_images,32,32,3))
	
	#~ no_images = 0
	
	#~ for l in range(10):
	    #~ counter = 0
	    #~ img_files = sorted(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/*'))
	    #~ no_images += len(img_files)
	
	#~ labels = np.zeros((no_images,)).astype(int)
	#~ images = np.zeros((no_images,32,32,3))
	
	#~ counter = 0
	
	for l in range(10):
	    
	    print l
	    counter = 0
	    
	    img_files = sorted(np.array(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/*')))
	    #~ values = np.array([float(v.split('_')[-1].split('.p')[0]) for v in img_files])
	    #~ img_files = img_files[np.argsort(values)[:no_images]]
	    img_files = img_files[:no_images]
	    
	    for img_dir in img_files:
		#~ print img_dir
		im = misc.imread(img_dir)
		im = np.expand_dims(im, axis=0)
		images[no_images * l + counter] = im
		labels[no_images * l + counter] = l
		counter+=1
	
	npr.seed(2301)
	random_idx = np.arange(len(images))
	npr.shuffle(random_idx)
	images = images[random_idx]
	labels = labels[random_idx]
	
	print 'break'
	images = images / 127.5 - 1
	return images, labels

    def load_gen_images(self):
	
	'''
	Loading images generated with eval_dsn()
	Assuming that ./sample contains folder with
	subfolders 1,2,...,9.
	'''
	
	print 'Loading generated images.'
	
	no_images = 0
	v_threshold = 12.0
	experiment = '0.01_0.01_5.0'
	
	for l in range(10):
	    counter = 0
	    img_files = sorted(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/'+experiment+'/*'))
	    print img_files[-1]
	    values = np.array([float(v.split('_')[-1].split('.p')[0]) for v in img_files])
	    no_images += len(values[values>=v_threshold])
	
	labels = np.zeros((no_images,)).astype(int)
	images = np.zeros((no_images,32,32,3))
	
	counter = 0
	
	for l in range(10):
	    
	    img_files = np.array(sorted(np.array(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/'+experiment+'/*'))))
	    values = np.array([float(v.split('_')[-1].split('.p')[0]) for v in img_files])
	    img_files = img_files[values >= v_threshold]
	    
	    for img_dir in img_files:
		#~ print img_dir
		im = misc.imread(img_dir)
		im = np.expand_dims(im, axis=0)
		images[counter] = im
		labels[counter] = l
		counter+=1
		
	    print l, counter 
	
	npr.seed(231)
	random_idx = np.arange(len(images))
	npr.shuffle(random_idx)
	images = images[random_idx]
	labels = labels[random_idx]
	
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
	    
            #~ print ('Loading pretrained model.')
            #~ variables_to_restore = slim.get_model_variables(scope='encoder')
            #~ restorer = tf.train.Saver(variables_to_restore)
            #~ restorer.restore(sess, self.pretrained_model)
	    
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
	
	if self.protocol=='syn_svhn':
	    source_images, source_labels = self.load_syn(self.syn_dir, split='train')
	    target_images, target_labels = self.load_svhn(self.svhn_dir, split='train')
	    target_images = target_images[self.start_img:self.end_img]
	    target_labels = target_labels[self.start_img:self.end_img]

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
	    restorer.restore(sess, self.test_model)
	    
	    #~ print ('Loading pretrained generator.')
	    #~ variables_to_restore = slim.get_model_variables(scope='generator')
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, self.test_model)
	    
	    #~ print ('Loading pretrained disc_g.')
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
	    
	    
	    label_gen = utils.one_hot(np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5]),10)
	    
	    for step in range(self.train_iter):
		
		if step%100==0:
		    npr.shuffle(target_images)
		
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
		
		
		#~ sess.run(model.E_train_op, feed_dict) 
		#~ sess.run(model.DE_train_op, feed_dict)
		
		sess.run(model.G_train_op, feed_dict)
		sess.run(model.DG_train_op, feed_dict) 
		sess.run(model.const_train_op, feed_dict)

		logits_E_real,logits_E_fake,logits_G_real,logits_G_fake = sess.run([model.logits_E_real,model.logits_E_fake,model.logits_G_real,model.logits_G_fake],feed_dict) 
		
		if (step+1) % 1000 == 0:
		    
		    summary, E, DE, G, DG, cnst = sess.run([model.summary_op, model.E_loss, model.DE_loss, model.G_loss, model.DG_loss, model.const_loss], feed_dict)
		    summary_writer.add_summary(summary, step)
		    print ('Step: [%d/%d] E: [%.6f] DE: [%.6f] G: [%.6f] DG: [%.6f] Const: [%.6f] E_real: [%.2f] E_fake: [%.2f] G_real: [%.2f] G_fake: [%.2f]' \
			       %(step+1, self.train_iter, E, DE, G, DG, cnst,logits_E_real.mean(),logits_E_fake.mean(),logits_G_real.mean(),logits_G_fake.mean()))

		    

		if (step+1) % 10000 == 0:
		    saver.save(sess, os.path.join(self.model_save_path, 'dtn'))

    def eval_dsn(self, name = '1600'):
        # build model
        model = self.model
        model.build_model()

	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
        with tf.Session(config=self.config) as sess:
	    
	    print ('Loading pretrained G.')
	    variables_to_restore = slim.get_model_variables(scope='generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.test_model)
	    
	    print ('Loading pretrained E.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.test_model)
	    
	    print ('Loading sample generator.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    
	    if self.protocol=='svhn_mnist':
		source_images, source_labels = self.load_svhn(self.svhn_dir)
	    elif self.protocol=='mnist_usps':
		source_images, source_labels = self.load_mnist(self.mnist_dir)
		
	    npr.seed(190)

	    for n in range(10):
		
		print n
	    
		no_gen = 10000

		source_labels = n * np.ones((no_gen,),dtype=int)

		# train model for source domain S
		src_labels = utils.one_hot(source_labels[:no_gen],10)
		src_noise = utils.sample_Z(no_gen,100,'uniform')

		feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels}

		samples, samples_logits = sess.run([model.sampled_images, model.sampled_images_logits], feed_dict)
		samples_logits = samples_logits[:,n]
		#~ samples = samples[samples_logits>8.]
		#~ samples_logits = samples_logits[samples_logits>8.]
		
		for i in range(len(samples_logits)):
		    try:
			imsave('./sample/'+str(np.argmax(src_labels[i]))+'/'+name+'/'+str(i)+'_'+str(np.argmax(src_labels[i]))+'_'+str(samples_logits[i])+'.png',np.squeeze(samples[i]))
		    except:
			os.mkdir('./sample/'+str(np.argmax(src_labels[i]))+'/'+name+'/')
			
		print str(i)+'/'+str(len(samples)), np.argmax(src_labels[i])
		    
    def train_gen_images(self):
        # load svhn dataset
        src_images, src_labels = self.load_gen_images()
	
	
	if self.protocol == 'svhn_mnist':
	    trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='test')
	if self.protocol == 'syn_svhn':
	    trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='test')
	elif self.protocol == 'mnist_usps':
	    trg_images, trg_labels = self.load_usps(self.usps_dir)
	elif self.protocol == 'usps_mnist':
	    trg_images, trg_labels = self.load_mnist(self.mnist_dir)
	
        # build a graph
        model = self.model
        model.build_model()
		
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
	    #~ print ('Loading pretrained encoder.')
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
		    
		    
		    if (t+1) % 100 == 0:
			#~ print 'Saved.'
			#~ saver.save(sess, os.path.join(self.model_save_path, 'model_gen'))
			
			summary, l, src_acc = sess.run([model.summary_op, model.loss, model.src_accuracy], feed_dict)
			src_rand_idxs = np.random.permutation(src_images.shape[0])[:1000]
			trg_rand_idxs = np.random.permutation(trg_images.shape[0])[:1000]
			summary, l, src_acc, test_acc = sess.run([model.summary_op, model.loss, model.src_accuracy, model.trg_accuracy], 
					       feed_dict={model.src_images: src_images[src_rand_idxs], 
							  model.src_labels: src_labels[src_rand_idxs],
							  model.trg_images: trg_images[trg_rand_idxs], 
							  model.trg_labels: trg_labels[trg_rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f] train acc: [%.3f] test acc [%.3f]' \
				   %(t+1, self.pretrain_iter, l, src_acc, test_acc))
			
    def check_TSNE(self):
	
	if self.protocol == 'svhn_mnist':
	    source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	    target_images, target_labels = self.load_mnist(self.mnist_dir, split='train')
	
	if self.protocol == 'mnist_mnist_m':
	    source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	    target_images, target_labels = self.load_mnist_m(self.mnist_m_dir, split='train')
	
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
	
	elif self.protocol == 'amazon_reviews':
	    source_images, source_labels, target_images, target_labels, _, _ = self.load_amazon_reviews(self.amazon_dir)
	    
	
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
	    
            
	    n_samples = 1000
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
		
		f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
		ax1.set_facecolor('white')
		ax2.set_facecolor('white')
		
		ax1.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)))), s=3, cmap = mpl.cm.jet, alpha=0.5)
		ax2.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels,src_labels)), s=3, cmap = mpl.cm.jet, alpha=0.5)


	    elif sys.argv[2] == '3':
		TSNE_hA = model.fit_transform(np.vstack((fzy,fx_src,fx_trg)))
		
		f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
		ax1.set_facecolor('white')
		ax2.set_facecolor('white')

		ax1.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)), 3 * np.ones((n_samples,)))), s=5,  cmap = mpl.cm.jet, alpha=0.5)
		ax2.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels, src_labels, trg_labels, )), s=5,  cmap = mpl.cm.jet, alpha=0.5)
		
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
	
	elif self.protocol == 'mnist_mnist_m':
	    
	    src_images, src_labels = self.load_mnist(self.mnist_dir, split='train')
	    src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')
	    
	    trg_images, trg_labels = self.load_mnist_m(self.mnist_m_dir, split='train')
	    trg_test_images, trg_test_labels = self.load_mnist_m(self.mnist_m_dir, split='test')
	
	elif self.protocol == 'syn_svhn':
	    
	    src_images, src_labels = self.load_syn(self.syn_dir, split='train')
	    src_test_images, src_test_labels = self.load_syn(self.syn_dir, split='test')
	    
	    trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='test')
	    trg_test_images, trg_test_labels = self.load_svhn(self.svhn_dir, split='test')
	
	elif self.protocol == 'mnist_usps':
	    
	    src_images, src_labels = self.load_mnist(self.mnist_dir, split='train')
	    src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')
	    
	    trg_images, trg_labels = self.load_usps(self.usps_dir)
	    trg_test_images = trg_images
	    trg_test_labels = trg_labels
	    
	elif self.protocol == 'usps_mnist':
	    
	    trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train')
	    trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')
	    
	    src_images, src_labels = self.load_usps(self.usps_dir)
	    src_test_images = src_images
	    src_test_labels = src_labels
	
	elif self.protocol == 'amazon_reviews':
	    src_images, src_labels, trg_images, trg_labels, trg_test_images, trg_test_labels = self.load_amazon_reviews(self.amazon_dir)
	    src_test_images, src_test_labels = src_images, src_labels
	
	
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
		
		elif sys.argv[1] == 'gen':
		    print ('Loading gen model.')
		    variables_to_restore = slim.get_model_variables(scope='encoder')
		    restorer = tf.train.Saver(variables_to_restore)
		    restorer.restore(sess, self.gen_model)
		    
		else:
		    raise NameError('Unrecognized mode.')
	    
		t+=1
    
		src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:3000]
		trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:]
		test_src_acc, test_trg_acc, trg_pred = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.trg_pred], 
				       feed_dict={model.src_images: src_test_images[src_rand_idxs], 
						  model.src_labels: src_test_labels[src_rand_idxs],
						  model.trg_images: trg_test_images[trg_rand_idxs], 
						  model.trg_labels: trg_test_labels[trg_rand_idxs]})
		src_acc = sess.run(model.src_accuracy, feed_dict={model.src_images: src_images[:3000], 
								  model.src_labels: src_labels[:3000],
						                  model.trg_images: trg_test_images[trg_rand_idxs], 
								  model.trg_labels: trg_test_labels[trg_rand_idxs]})
						  
		print ('Step: [%d/%d] src train acc [%.3f]  src test acc [%.3f] trg test acc [%.3f]' \
			   %(t+1, self.pretrain_iter, src_acc, test_src_acc, test_trg_acc))
		
		print confusion_matrix(trg_test_labels[trg_rand_idxs], trg_pred)	   
		
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
	
		time.sleep(10.1)
		    
if __name__=='__main__':

    from model import DSN
    model = DSN(mode='eval_dsn')
    solver = Solver(model)
    solver.check_TSNE()




