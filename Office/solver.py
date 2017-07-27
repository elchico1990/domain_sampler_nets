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
from PIL import Image

import time
import glob 

import matplotlib.pyplot as plt
import matplotlib as mpl

import utils

from sklearn.manifold import TSNE

class Solver(object):

    def __init__(self, model, batch_size=64, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 src_dir='webcam', trg_dir='dslr', log_dir='logs', sample_save_path='sample', 
                 model_save_path='model', pretrained_model='model/model', pretrained_sampler='model/sampler', 
		 test_model='model/dtn', convdeconv_model = 'model/conv_deconv'):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.src_dir = src_dir
        self.trg_dir = trg_dir	
	self.base_path = src_dir+'2'+trg_dir+'/'
        self.log_dir = self.base_path+log_dir
        self.sample_save_path = self.base_path+sample_save_path
        self.model_save_path = self.base_path+model_save_path
        self.pretrained_model = self.base_path+pretrained_model
	self.pretrained_sampler = self.base_path+pretrained_sampler
        self.test_model = self.base_path+test_model
	self.convdeconv_model = self.base_path+convdeconv_model
	    # create directories if not exist
	if not tf.gfile.Exists(self.model_save_path):
	    tf.gfile.MakeDirs(self.model_save_path)
	if not tf.gfile.Exists(self.sample_save_path):
	    tf.gfile.MakeDirs(self.sample_save_path)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	
	self.user = 'Ric' # to load the dataset.
	
	self.no_images = {'amazon':2817, 'dslr':498, 'webcam':795}

    def load_office(self, split, image_dir='./office'):
        print ('Loading OFFICE dataset -> '+split)

	if self.user == 'Pie':
	    if split == 'amazon':
		image_file1 = 'amazon_1.pkl'
		image_file2 = 'amazon_2.pkl' 
		image_dir1 = os.path.join(image_dir, image_file1)
		image_dir2 = os.path.join(image_dir, image_file2)
		with open(image_dir1, 'rb') as f:
		    office = pickle.load(f)
		    images = office['X']
		    labels = office['y']
		with open(image_dir2, 'rb') as f:
		    office = pickle.load(f)
		    images = np.concatenate([images, office['X']], axis=0)
		    labels = np.concatenate([labels, office['y']], axis=0)
		    
	    else:
		image_file = split+'.pkl' 
		image_dir = os.path.join(image_dir, image_file)
		with open(image_dir, 'rb') as f:
		    office = pickle.load(f)
		    images = office['X']
		    labels = office['y']
	
	elif self.user == 'Ric':
	    images = np.zeros((self.no_images[split],227,227,3))
	    labels = np.zeros((self.no_images[split],1))
	    l = 0
	    c = 0
	    obj_categories = sorted(glob.glob(image_dir + '/' + split + '/images/*'))
	    for oc in obj_categories:
		obj_images = sorted(glob.glob(oc+'/*'))
		#~ print str(l)+'/'+str(len(obj_categories))
		for oi in obj_images:
		    img = Image.open(oi)
		    img = img.resize((227,227), Image.ANTIALIAS) - np.array([104., 117., 124.])
		    img = np.expand_dims(img, axis=0)
		    images[c] = img
		    labels[c] = l
		    c+=1
	    	l+=1
		
	rnd_indices = np.arange(len(labels))
	npr.shuffle(rnd_indices)
	images = images[rnd_indices]
	labels = labels[rnd_indices]
        return images, np.squeeze(labels)

    def pretrain(self):
        src_images, src_labels = self.load_office(split=self.src_dir)
        trg_images, trg_labels = self.load_office(split=self.trg_dir)
	        

        # build a graph
        model = self.model
        model.build_model()
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
	    saver = tf.train.Saver()
	    
	    model.model_AlexNet.load_initial_weights(sess)
	    
            #~ print ('Loading pretrained model.')
            #~ variables_to_restore = slim.get_model_variables(scope='encoder')
            #~ restorer = tf.train.Saver(variables_to_restore)
            #~ restorer.restore(sess, self.pretrained_model)
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    epochs = 300
	    
	    t = 0
	    
	    self.batch_size = 32

	    for i in range(epochs):
		
		print 'Epoch',str(i)
		src_rand = np.random.permutation(src_images.shape[0])
		src_images, src_labels = src_images[src_rand], src_labels[src_rand]
		
		for start, end in zip(range(0, len(src_images), self.batch_size), range(self.batch_size, len(src_images), self.batch_size)):
		    
		    t+=1
		       
		    feed_dict = {model.keep_prob : 1.0, model.src_images: src_images[start:end], model.src_labels: src_labels[start:end], 
						model.trg_images: trg_images[0:2], model.trg_labels: trg_labels[0:2]} #trg here is just needed by the model but actually useless. 
		    
		    sess.run(model.train_op, feed_dict)
		    
		    #~ qwe, asd = sess.run([model.src_pred, model.src_labels],feed_dict) 
		    #~ print 'break'


		summary, l = sess.run([model.summary_op, model.loss], feed_dict)
		src_rand_idxs = np.random.permutation(src_images.shape[0])[:100]
		trg_rand_idxs = np.random.permutation(trg_images.shape[0])[:100]
		src_acc, trg_acc = sess.run(fetches=[model.src_accuracy, model.trg_accuracy], 
				       feed_dict={model.keep_prob : 1.0,
						    model.src_images: src_images[src_rand_idxs], 
						    model.src_labels: src_labels[src_rand_idxs],
						    model.trg_images: trg_images[trg_rand_idxs], 
						    model.trg_labels: trg_labels[trg_rand_idxs]})
		summary_writer.add_summary(summary, t)
		print ('Step: [%d/%d] loss: [%.6f]  src acc [%.2f] trg acc [%.2f]' \
			   %(t+1, self.pretrain_iter, l, src_acc, trg_acc))
		
		#~ # 'Saved.'
		saver.save(sess, os.path.join(self.model_save_path, 'model'))
    
    def train_sampler(self):
	
	print 'Training sampler.'
        # load svhn dataset
        source_images, source_labels = self.load_svhn(self.src_dir, split='train')
	source_labels = utils.one_hot(source_labels, 10)
	
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
	epochs = 50

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
        
	target_images, target_labels = self.load_mnist(self.trg_dir, split='train')
	#~ usps_images, usps_labels = self.load_usps(self.usps_dir)
	source_images, source_labels = self.load_svhn(self.src_dir, split='train')
	

        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
	    with tf.device('/gpu:2'):
			    
		# initialize G and D
		tf.global_variables_initializer().run()
		# restore variables of F
		
		print ('Loading pretrained encoder.')
		variables_to_restore = slim.get_model_variables(scope='encoder')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.test_model)
		
		print ('Loading pretrained encoder disc.')
		variables_to_restore = slim.get_model_variables(scope='disc_e')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.pretrained_sampler)
		
		print ('Loading pretrained G.')
		variables_to_restore = slim.get_model_variables(scope='generator')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.test_model)
		
		print ('Loading pretrained D_g.')
		variables_to_restore = slim.get_model_variables(scope='disc_g')
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
		
		G_loss = 1.
		DG_loss = 1.
		
		label_gen = utils.one_hot(np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]),10)
		
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
		    
		    #~ sess.run(model.E_train_op, feed_dict) 
		    #~ sess.run(model.DE_train_op, feed_dict) 
		    
		    sess.run(model.G_train_op, feed_dict) 
		    sess.run(model.DG_train_op, feed_dict) 
		    
		    sess.run(model.const_train_op, feed_dict)
		    
		    logits_E_real,logits_E_fake,logits_G_real,logits_G_fake = sess.run([model.logits_E_real,model.logits_E_fake,model.logits_G_real,model.logits_G_fake],feed_dict) 
		    
		    if (step+1) % 10 == 0:
			
			summary, E, DE, G, DG, cnst = sess.run([model.summary_op, model.E_loss, model.DE_loss, model.G_loss, model.DG_loss, model.const_loss], feed_dict)
			summary_writer.add_summary(summary, step)
			print ('Step: [%d/%d] E: [%.6f] DE: [%.6f] G: [%.6f] DG: [%.6f] Const: [%.6f] E_real: [%.2f] E_fake: [%.2f] G_real: [%.2f] G_fake: [%.2f]' \
				   %(step+1, self.train_iter, E, DE, G, DG, cnst,logits_E_real.mean(),logits_E_fake.mean(),logits_G_real.mean(),logits_G_fake.mean()))

			

		    if (step+1) % 500 == 0:
			saver.save(sess, os.path.join(self.model_save_path, 'dtn'))
            
    def eval_dsn(self):
        # build model
        model = self.model
        model.build_model()

        # load svhn dataset
        source_images, source_labels = self.load_svhn(self.src_dir)
	source_labels[:] = 2

        with tf.Session(config=self.config) as sess:
	    
	    
	
	    
	    print ('Loading pretrained G.')
	    variables_to_restore = slim.get_model_variables(scope='generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.test_model)
	    
	    
	    print ('Loading sample generator.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    


	    # train model for source domain S
	    src_labels = utils.one_hot(source_labels[:1000],10)
	    src_noise = utils.sample_Z(1000,100,'uniform')

	    feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels}

	    samples = sess.run(model.sampled_images, feed_dict)

	    for i in range(1000):
		
		print str(i)+'/'+str(len(samples)), np.argmax(src_labels[i])
		plt.imshow(np.squeeze(samples[i]), cmap='gray')
		plt.imsave('./sample/'+str(np.argmax(src_labels[i]))+'/'+str(i)+'_'+str(np.argmax(src_labels[i])),np.squeeze(samples[i]), cmap='gray')

    def check_TSNE(self):
	
	target_images, target_labels = self.load_mnist(self.trg_dir, split='train')
	#~ usps_images, usps_labels = self.load_usps(self.usps_dir)
	source_images, source_labels = self.load_svhn(self.src_dir, split='train')
	

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
		plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels, src_labels, trg_labels, )), s=3,  cmap = mpl.cm.jet)
	        plt.figure(3)
                plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)), 3 * np.ones((n_samples,)))), s=3,  cmap = mpl.cm.jet)

	    elif sys.argv[2] == '4':
		TSNE_hA = model.fit_transform(h_repr)
	        plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.argmax(trg_labels,1), s=3,  cmap = mpl.cm.jet)
		

	    plt.show()
	    
    def test(self):
	
	# load svhn dataset
	src_images, src_labels = self.load_office(split=self.src_dir)
        trg_images, trg_labels = self.load_office(split=self.trg_dir)
	
	
	# build a graph
	model = self.model
	model.build_model()
		
	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
	with tf.Session(config=self.config) as sess:
	    tf.global_variables_initializer().run()
	    saver = tf.train.Saver()
	    
	    t = 0
	    
	    while(True):
		
		if sys.argv[1] == 'test':
		    print ('Loading test model.')
		    variables_to_restore = tf.global_variables()
		    restorer = tf.train.Saver(variables_to_restore)
		    restorer.restore(sess, self.test_model)
		
		elif sys.argv[1] == 'pretrain':
		    print ('Loading pretrained model.')
		    variables_to_restore = tf.global_variables()
		    restorer = tf.train.Saver(variables_to_restore)
		    restorer.restore(sess, self.pretrained_model)
		    
		else:
		    raise NameError('Unrecognized mode.')
	    
		t+=1
    
		src_acc, trg_acc, _ = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.loss], 
				       feed_dict={model.src_images: src_images, 
						  model.src_labels: src_labels,
						  model.trg_images: trg_images, 
						  model.trg_labels: trg_labels})
		  
		print ('Step: [%d/%d] src acc [%.2f] trg acc [%.2f]' \
			   %(t+1, self.pretrain_iter, src_acc, trg_acc))
	
		time.sleep(.5)
		    
if __name__=='__main__':

    from model import DSN
    model = DSN(mode='eval_dsn', learning_rate=0.0003)
    solver = Solver(model)
    solver.check_TSNE()

