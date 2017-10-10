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

from scipy.misc import imsave

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import utils

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from scipy import misc

class Solver(object):

    def __init__(self, model, batch_size=16, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 svhn_dir='svhn', syn_dir='syn', mnist_dir='mnist', mnist_m_dir='mnist_m', usps_dir='usps', amazon_dir='amazon_reviews', cifar_dir='cifar10',
		 log_dir='logs', sample_save_path='sample', model_save_path='model', pretrained_model='model/model', gen_model='model/model_gen', pretrained_sampler='model/sampler', 
		 test_model='model/dtn', convdeconv_model = 'model/conv_deconv', start_img = 0, end_img = 0):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.cifar_dir = '/home/rvolpi/Desktop/domain_sampler_nets/dtn/data/'+cifar_dir
        self.svhn_dir = 'data/'+svhn_dir
        self.syn_dir = 'data/'+syn_dir
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
	self.protocol = 'cifar_stl' # possibilities: svhn_mnist, mnist_usps, mnist_usps_2, usps_mnist, syn_svhn, mnist_mnist_m, amazon_reviews, stl_cifar, cifar_stl
	
	self.start_img = start_img
	self.end_img = end_img
	
    def load_cifar(self,split = 'train'):
	
	print 'Loading CIFAR-10'
	trX = np.zeros((50000,3072))
	trY = np.zeros((50000,))
	
	for i in range(1,6):
	    f = utils.unpickle(self.cifar_dir+'/data_batch_'+str(i))
	    
	    trX[(i-1)*10000:i*10000] = f['data']
		    
	    trY[(i-1)*10000:i*10000] = f['labels']

	f = utils.unpickle(self.cifar_dir+'/test_batch')
	
	teX = f['data']
	teY = f['labels']
	
	trX = trX.reshape(50000,32,32,3)
	teX = teX.reshape(10000,32,32,3)
	
	trX = trX/127.5 - 1
	teX = teX/127.5 - 1
	
	if split == 'train':
	    return trX,np.array(trY)
	else: 
	    return teX,np.array(teY)	

    def load_gen_images(self):
	
	'''
	Loading images generated with eval_dsn()
	Assuming that ./sample contains folder with
	subfolders 1,2,...,9.
	'''
	
	print 'Loading generated images.'
	
	no_images = 0
	v_threshold = 8.0
	experiment = '0.01_0.01_5.0'
	
	for l in range(10):
	    counter = 0
	    img_files = sorted(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/*'))
	    print img_files[-1]
	    values = np.array([float(v.split('_')[-1].split('.p')[0]) for v in img_files])
	    no_images += len(values[values>=v_threshold])
	
	labels = np.zeros((no_images,)).astype(int)
	images = np.zeros((no_images,32,32,3))
	
	counter = 0
	
	for l in range(10):
	    
	    img_files = np.array(sorted(np.array(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/*'))))
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
        
	src_images, src_labels = self.load_cifar(split='train')
	src_test_images, src_test_labels = self.load_cifar(split='test')
        trg_images, trg_labels = self.load_cifar(split='train')
	trg_test_images, trg_test_labels = self.load_cifar(split='test')

	
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

		    if (t+1) % 1000 == 0:
			summary, l, src_acc = sess.run([model.summary_op, model.loss, model.src_accuracy], feed_dict)
			src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:1000]
			trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:1000]
			test_src_acc, test_trg_acc, _ = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.loss], 
					       feed_dict={model.src_images: src_test_images[src_rand_idxs], 
							  model.src_labels: src_test_labels[src_rand_idxs],
							  model.trg_images: trg_test_images[trg_rand_idxs], 
							  model.trg_labels: trg_test_labels[trg_rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] src test acc [%.2f] trg test acc [%.4f]' \
				   %(t+1, self.pretrain_iter, l, src_acc, test_src_acc, test_trg_acc))
			
		    if (t+1) % 1000 == 0:
			#~ print 'Saved.'
			saver.save(sess, os.path.join(self.model_save_path, 'model'))
    
    def train_sampler(self):
	
	print 'Training sampler.'
        
	if self.protocol == 'svhn_mnist':	
	    source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	    source_labels = utils.one_hot(source_labels, 10)
        
	if self.protocol == 'mnist_mnist_m':	
	    source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	    source_labels = utils.one_hot(source_labels, 10)
        
	elif self.protocol == 'syn_svhn':	
	    source_images, source_labels = self.load_syn(self.syn_dir, split='train')
	    source_labels = utils.one_hot(source_labels, 10)

	elif self.protocol == 'mnist_usps':	
	    source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	    source_labels = utils.one_hot(source_labels, 10)
	    source_images = source_images[:2000]
	    source_labels = source_labels[:2000]
				  
	elif self.protocol == 'amazon_reviews':
	    source_images, source_labels, _, _, _, _ = self.load_amazon_reviews(self.amazon_dir)
	    source_labels = utils.one_hot(source_labels, 2)
	
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
			print ('Step: [%d/%d] d_loss: %.6f g_loss: %.6f avg_D_fake: %.2f avg_D_real: %.2f ' \
				   %(t+1, int(epochs*len(source_images) /batch_size), dl, gl, avg_D_fake.mean(), avg_D_real.mean()))
			
                    if (t+1) % 1000 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'sampler')) 

    def train_dsn(self):
        
	print 'Training DSN.'
	
	source_images, source_labels = self.load_syn(self.syn_dir, split='train')
	target_images, target_labels = self.load_svhn(self.svhn_dir, split='train')
	#~ target_images = target_images[self.start_img:self.end_img] 
	#~ target_labels = target_labels[self.start_img:self.end_img] 
	
	
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
	    
	    print ('Loading test encoder.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.test_model)
	    
	    print ('Loading sample generator.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    

	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
	    saver = tf.train.Saver()

	    print ('Start training.')
	    trg_count = 0
	    t = 0
	    
	    
	    label_gen = utils.one_hot(np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,9,9,9,9]),10)
	    
	    for step in range(45000):
		
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
		
		#~ sess.run(model.G_train_op, feed_dict)
		#~ sess.run(model.DG_train_op, feed_dict) 
		sess.run(model.const_train_op, feed_dict)
		
		#~ logits_E_real,logits_E_fake = sess.run([model.logits_E_real,model.logits_E_fake],feed_dict) 
		
		#~ if (step+1) % 100 == 0:
		    
		    #~ summary, E, DE, const_loss = sess.run([model.summary_op, model.E_loss, model.DE_loss, model.const_loss], feed_dict)
		    #~ summary_writer.add_summary(summary, step)
		    #~ print ('Step: [%d/%d] E: [%.3f] DE: [%.3f] const: [%.3f] E_real: [%.2f] E_fake: [%.2f]' \
			       #~ %(step+1, self.train_iter, E, DE, const_loss, logits_E_real.mean(),logits_E_fake.mean()))

		logits_G_real,logits_G_fake = sess.run([model.logits_G_real,model.logits_G_fake],feed_dict) 
		
		if (step+1) % 100 == 0:
		    
		    summary, G, DG, const_loss = sess.run([model.summary_op, model.G_loss, model.DG_loss, model.const_loss], feed_dict)
		    summary_writer.add_summary(summary, step)
		    print ('Step: [%d/%d] G: [%.3f] DG: [%.3f] const: [%.3f] G_real: [%.2f] G_fake: [%.2f]' \
			       %(step+1, self.train_iter, G, DG, const_loss, logits_G_real.mean(),logits_G_fake.mean()))

		
		
		
		    

		if (step+1) % 200 == 0:
		    saver.save(sess, os.path.join(self.model_save_path, 'dtn'))

    def eval_dsn(self, name = 'Exp2'):
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
	    
	    source_images, source_labels = self.load_svhn(self.svhn_dir)
	    
	    npr.seed(190)

	    for n in range(10):
		
		print n
	    
		no_gen = 5000

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
		    #~ try:
			#~ imsave('./sample/'+str(np.argmax(src_labels[i]))+'/'+name+'/'+str(i)+'_'+str(np.argmax(src_labels[i]))+'_'+str(samples_logits[i])+'.png',np.squeeze(samples[i]))
		    #~ except:
			#~ os.mkdir('./sample/'+str(np.argmax(src_labels[i]))+'/'+name+'/')
		    imsave('./sample/'+str(np.argmax(src_labels[i]))+'/'+name+'_'+str(i)+'_'+str(np.argmax(src_labels[i]))+'_'+str(samples_logits[i])+'.png',np.squeeze(samples[i]))
		    
		print str(i)+'/'+str(len(samples)), np.argmax(src_labels[i])

    def train_gen_images(self):
        # load svhn dataset
        src_images, src_labels = self.load_gen_images()
	trg_images, trg_labels = self.load_svhn(self.svhn_dir,split='train')
	
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
		    
		    if (t+1) % 1000 == 0:
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
			
		    if (t+1) % 1000 == 0:
			#~ print 'Saved.'
			saver.save(sess, os.path.join(self.model_save_path, 'model_gen'))
			
    def check_TSNE(self):
	
	source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	target_images, target_labels = self.load_usps(self.usps_dir, split='train')
	
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
	    
            
	    n_samples = 2000
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
	
	src_images, src_labels = self.load_syn(self.syn_dir, split='train')
	src_test_images, src_test_labels = self.load_syn(self.syn_dir, split='test')
	
	trg_images, trg_labels = self.load_svhn(self.svhn_dir, split='train')
	trg_test_images, trg_test_labels = self.load_svhn(self.svhn_dir, split='test')
    
	
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
    
		src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:]
		trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:]
		test_src_acc, test_trg_acc, trg_pred = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.trg_pred], 
				       feed_dict={model.src_images: src_test_images[src_rand_idxs], 
						  model.src_labels: src_test_labels[src_rand_idxs],
						  model.trg_images: trg_test_images[trg_rand_idxs], 
						  model.trg_labels: trg_test_labels[trg_rand_idxs]})
		src_acc = sess.run(model.src_accuracy, feed_dict={model.src_images: src_images[:1000], 
								  model.src_labels: src_labels[:1000],
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
	
		time.sleep(60.0)

    def find_closest_samples(self,dataset='MNIST'):
	
	img_dirs = ['./sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/0/22_0_10.3987',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/0/70_0_10.2846',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/0/93_0_10.3288',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/0/83_0_11.7384',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/0/107_0_11.6718',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/0/122_0_10.4151',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/0/322_0_10.3692',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/1/83_1_10.5441',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/1/203_1_10.3941',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/1/340_1_10.6729',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/2/101_2_10.2507',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/2/172_2_11.3251',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/22_3_14.1982',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/284_3_10.9449',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/193_3_13.2846',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/292_3_15.0359',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/311_3_11.4433',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/366_3_13.6956',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/779_3_14.2253',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/844_3_12.6912',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/1294_3_13.9789',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/1423_3_14.7053',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/3/1602_3_13.9868',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/32_4_11.5346',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/188_4_10.5293',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/211_4_12.0768',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/261_4_11.849',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/392_4_12.0682',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/572_4_11.1724',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/1001_4_12.8955', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/4/1103_4_12.0236', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/4_5_11.837',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/553_5_12.5355',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/730_5_11.5124',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/794_5_10.2816',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/1640_5_11.283',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/1933_5_10.2202', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/2400_5_10.458',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/4077_5_12.1508',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/5/4451_5_10.3354',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/357_6_14.8954',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/373_6_12.615', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/340_6_11.0551', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/633_6_14.9426',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/771_6_14.8695',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/939_6_12.1435',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/1100_6_13.7433',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/6/1265_6_13.1202',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/55_7_12.3649',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/330_7_10.1434',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/415_7_11.6358',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/385_7_12.822',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/927_7_11.3426',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/991_7_13.489',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/2594_7_11.6667',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/3199_7_10.4819',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/3271_7_12.9551', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/3503_7_10.5854',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/3642_7_11.7087', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/3981_7_11.6662',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/7664_7_10.603',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/10082_7_12.4001',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/7/17551_7_12.8428',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/8/73_8_13.8413',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/8/0_8_12.1716',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/8/362_8_13.1533',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/8/740_8_14.3074', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/8/742_8_13.8002', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/8/779_8_12.1449', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/50_9_12.1495',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/52_9_10.6145',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/62_9_11.5245', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/201_9_10.1126', #nice
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/206_9_10.7436',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/217_9_10.7002',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/1257_9_10.9516',
		    './sample/SVHN_MNIST_generated_88.1scratch_from60_commit_2c2ea5329bb8c8c3d5552ec14071435117925359/9/3028_9_10.884']
	
	img_dirs=['/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/0/34_0_12.5879',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/0/43_0_12.7191',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/0/71_0_13.4629',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/0/107_0_13.852',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/0/156_0_9.50515',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/0/179_0_12.3553',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/1/0_1_11.0697',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/1/39_1_11.906',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/1/47_1_12.02',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/1/260_1_11.8137',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/1/220_1_11.2743',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/2/16_2_14.8628',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/2/36_2_14.227',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/2/153_2_15.0372',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/2/205_2_11.1629',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/2/210_2_14.0031',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/2/211_2_14.932',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/2/232_2_14.8322',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/3/1_3_13.3803',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/3/205_3_13.1647',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/3/252_3_11.8634',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/3/457_3_12.8652',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/3/643_3_12.3128',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/4/289_4_12.7546',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/4/361_4_11.6547',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/4/468_4_13.9029',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/4/1316_4_13.5783',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/5/643_5_13.8028',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/5/816_5_11.4797',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/5/887_5_12.926',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/5/1021_5_13.1642',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/6/2_6_14.4633',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/6/20_6_14.6705',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/6/154_6_14.5059',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/6/733_6_14.5222',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/7/3_7_12.8828',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/7/46_7_14.3591',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/7/80_7_12.8622',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/7/146_7_14.6746',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/8/7_8_12.9503',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/8/112_8_13.1749',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/8/128_8_12.1496',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/8/119_8_13.9465',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/9/24_9_13.2987',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/9/40_9_14.0222',
		'/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/MNIST_USPS_91.7_commit_830e57b74af6600af0c00d8fc9c92a03668d25dd/9/57_9_13.7739']
		    
	
	#~ real_images, real_labels = self.load_mnist(self.mnist_dir, split='train')
	real_images, real_labels = self.load_usps(self.usps_dir)
	real_images = real_images[:1800]
	real_labels = real_labels[:1800]

	
	counter=0
	for img_dir in img_dirs:
	    counter+=1
	    print counter,'/',str(len(img_dirs))
	    gen_image = misc.imread(img_dir, mode='L')
	    gen_image = np.expand_dims(gen_image, axis=0)
	    gen_image = np.expand_dims(gen_image, axis=3)
	    gen_image = gen_image/ 127.5 - 1

	    diff_square = np.sum(np.square(np.squeeze(real_images-gen_image).reshape((len(real_images), 28 * 28))),1)
	    closest_image = real_images[np.argmin(diff_square)]

	    gen_image = np.squeeze(gen_image)
	    closest_image = np.squeeze(closest_image)

	    plt.imsave('./images_for_paper/'+str(counter)+'_generated.png',gen_image,cmap='gray')
	    plt.imsave('./images_for_paper/'+str(counter)+'_closest.png',closest_image,cmap='gray')


		
if __name__=='__main__':
    
    
    

    from model import DSN
    model = DSN(mode='eval_dsn', learning_rate=0.0003)
    solver = Solver(model)
    solver.find_closest_samples()
    
    #~ solver.check_TSNE()