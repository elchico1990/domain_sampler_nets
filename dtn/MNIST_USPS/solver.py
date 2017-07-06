import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
import cPickle

import time

import matplotlib.pyplot as plt
import matplotlib as mpl

import utils

from sklearn.manifold import TSNE

class Solver(object):

    def __init__(self, model, batch_size=64, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 svhn_dir='svhn', mnist_dir='mnist', usps_dir='usps', log_dir='logs', sample_save_path='sample', 
                 model_save_path='model', pretrained_model='model/model', pretrained_sampler='model/sampler', 
		 test_model='model/dtn'):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.svhn_dir = svhn_dir
        self.mnist_dir = mnist_dir
        self.usps_dir = usps_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
	self.pretrained_sampler = pretrained_sampler
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True

    def load_svhn(self, image_dir, split='train'):
        print ('Loading SVHN dataset.')
        
        if self.model.mode in ['pretrain']:
            image_file = 'extra_32x32.mat' if split=='train' else 'test_32x32.mat'
        else:
            image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
        labels = svhn['y'].reshape(-1)
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
	
    def load_office(image_dir='./office', split='amazon'):
        print ('Loading OFFICE dataset ->'+split)

	
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
		
        return images, labels

    def load_usps(self, image_dir):
        
	print ('Loading USPS dataset.')
        image_file = 'train.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            usps = pickle.load(f)
        images = usps['X'] / 127.5 - 1
        labels = usps['y']
        return images, labels
	
    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        return merged

    def pretrain(self):
        # load svhn dataset
        src_images, src_labels = self.load_mnist(self.mnist_dir, split='train')
        src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')

        trg_images, trg_labels = self.load_usps(self.usps_dir)
        trg_test_images, trg_test_labels = self.load_usps(self.usps_dir)
	trg_labels = np.squeeze(trg_labels.T)
	trg_test_labels = np.squeeze(trg_test_labels.T)
	trg_labels-=1
	trg_labels[trg_labels==255] = 9
	trg_test_labels-=1
	trg_test_labels[trg_labels==255] = 9
	
	#~ src_images = src_images[2000:4000]
	#~ src_labels = src_labels[2000:4000]
	#~ src_test_images = src_test_images[2000:4000]
	#~ src_test_labels = src_test_labels[2000:4000]
	
	
	#~ trg_images = trg_images[2000:3800]
	#~ trg_labels = trg_labels[2000:3800]
	#~ trg_test_images = trg_test_images[2000:3800]
	#~ trg_test_labels = trg_test_labels[2000:3800]
	
	#~ for i in range(80):
	    #~ print trg_labels[i]
	    #~ plt.imshow(np.squeeze(trg_images[i]))
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

	    epochs = 1000
	    
	    t = 0

	    for i in range(epochs):
		
		#~ print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(src_images), self.batch_size), range(self.batch_size, len(src_images), self.batch_size)):
		    
		    t+=1
		    
		    
		    feed_dict = {model.src_images: src_images[start:end], model.src_labels: src_labels[start:end], model.trg_images: trg_images[:3], model.trg_labels: trg_labels[:3]}
		    
		    sess.run(model.train_op, feed_dict) 

		    if (t+1) % 100 == 0:
			summary, l, src_acc = sess.run([model.summary_op, model.loss, model.src_accuracy], feed_dict)
			src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:1000]
			trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:1000]
			test_src_acc, test_trg_acc, _ = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.loss], 
					       feed_dict={model.src_images: src_test_images[src_rand_idxs], 
							  model.src_labels: src_test_labels[src_rand_idxs],
							  model.trg_images: trg_test_images[trg_rand_idxs], 
							  model.trg_labels: trg_test_labels[trg_rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] src test acc [%.2f] trg test acc [%.2f]' \
				   %(t+1, self.pretrain_iter, l, src_acc, test_src_acc, test_trg_acc))
			
		    if (t+1) % 100 == 0:
			#~ print 'Saved.'
			saver.save(sess, os.path.join(self.model_save_path, 'model'))

    def train_classifier(self):
        # load svhn dataset
        src_images, src_labels = self.load_mnist(self.mnist_dir, split='train')
        src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')
        
        trg_images, trg_labels = self.load_usps(self.usps_dir)
        trg_labels = np.squeeze(trg_labels.T)
	trg_labels-=1
	trg_labels[trg_labels==255] = 9
	
	trg_test_images = trg_images
	trg_test_labels = trg_labels
	
	src_labels = utils.one_hot(src_labels,10)
	trg_labels = utils.one_hot(trg_labels,10)
	src_test_labels = utils.one_hot(src_test_labels,10)
	trg_test_labels = utils.one_hot(trg_test_labels,10)
	
        # build a graph
        model = self.model
        model.build_model()
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            print ('Loading E.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.test_model)
	    
            print ('Loading sampler.')
            variables_to_restore = slim.get_model_variables(scope='sampler_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)
	    
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    epochs = 100
	    noise_dim = 100
	
	    
	    t = 0

	    for i in range(epochs):
		
		#~ print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(src_images), self.batch_size), range(self.batch_size, len(src_images), self.batch_size)):
		    
		    t+=1
		    
		    Z_samples = utils.sample_Z(self.batch_size, noise_dim, 'uniform')
		       
		    feed_dict = {model.src_noise: Z_samples, model.src_images: src_images[start:end], model.src_labels: src_labels[start:end], model.trg_images: trg_images[:3], model.trg_labels: trg_labels[:3]}
		    
		    sess.run(model.train_op, feed_dict) 

		    if (t+1) % 500 == 0:
			summary, l = sess.run([model.summary_op, model.loss], feed_dict)
			src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:1000]
			trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:4000]
			fzy_acc, fx_src_acc, fx_trg_acc, _ = sess.run(fetches=[model.fzy_accuracy, model.fx_src_accuracy, model.fx_trg_accuracy, model.loss], 
					       feed_dict={model.src_noise: utils.sample_Z(1000, noise_dim, 'uniform'),
							  model.src_images: src_test_images[src_rand_idxs], 
							  model.src_labels: src_test_labels[src_rand_idxs],
							  model.trg_images: trg_test_images[trg_rand_idxs], 
							  model.trg_labels: trg_test_labels[trg_rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] src test acc [%.2f] trg test acc [%.2f]' \
				   %(t+1, self.pretrain_iter, l, fzy_acc, fx_src_acc, fx_trg_acc))
			
		    if (t+1) % 500 == 0:
			#~ print 'Saved.'
			saver.save(sess, os.path.join(self.model_save_path, 'model'))
	    
    def train_sampler(self):
	
	print 'Training sampler.'
        # load svhn dataset
        source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	source_labels = utils.one_hot(source_labels, 10)
	
	#~ source_images = source_images[2000:4000]
	#~ source_labels = source_labels[2000:4000]
	
        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	
	batch_size = 32
	noise_dim = 100
	epochs = 3000000

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
	    
            print ('Loading pretrained model.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
	    
	    #~ print ('Loading pretrained encoder disc.')
	    #~ variables_to_restore = slim.get_model_variables(scope='disc_e')
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, self.pretrained_sampler)
	    
	    #~ print ('Loading sample generator.')
	    #~ variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, self.pretrained_sampler)
	
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
		    
		    #~ a,b,c,d = sess.run([model.logits_fake,model.logits_real,model.labels_fake,model.labels_real], feed_dict)
		    
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
        
	#~ usps_images, usps_labels = self.load_usps(self.usps_dir)
	source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	target_images, target_labels = self.load_usps(self.usps_dir)
	
	#~ source_images = source_images[2000:4000]
	#~ source_labels = source_labels[2000:4000]
	#~ target_images = target_images[2000:3800]
	#~ target_labels = target_labels[2000:3800]
	
	
	
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
		restorer.restore(sess, self.pretrained_model)
		
		print ('Loading pretrained encoder disc.')
		variables_to_restore = slim.get_model_variables(scope='disc_e')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.pretrained_sampler)
		
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
		
		self.train_iter = 10000000
		
		#Inferring labels using pretrained E
		feed_dict = {model.src_images: source_images[:self.batch_size], model.src_noise: utils.sample_Z(self.batch_size,100,'uniform'), model.src_labels: utils.one_hot(source_labels[:self.batch_size],10),model.trg_labels: utils.one_hot(target_labels[:self.batch_size],10), model.trg_images: target_images}
		target_labels = sess.run(model.trg_labels_oh, feed_dict) 
		    
		print 'break'
		for step in range(self.train_iter+1):
		    
		    trg_count += 1
		    t+=1
		    
		    i = step % int(source_images.shape[0] / self.batch_size)
		    j = step % int(target_images.shape[0] / self.batch_size)
		    
		    
		    
		    src_images = source_images[i*self.batch_size:(i+1)*self.batch_size]
		    src_labels = utils.one_hot(source_labels[i*self.batch_size:(i+1)*self.batch_size],10)
		    src_labels_int = source_labels[i*self.batch_size:(i+1)*self.batch_size]
		    src_noise = utils.sample_Z(self.batch_size,100,'uniform')
		    trg_images = target_images[j*self.batch_size:(j+1)*self.batch_size]
		    trg_labels = target_labels[j*self.batch_size:(j+1)*self.batch_size]
		    
		    feed_dict = {model.src_images: src_images, model.src_noise: src_noise, model.src_labels: src_labels, model.trg_images: trg_images, model.trg_labels: trg_labels, }
		    
		    sess.run(model.E_train_op, feed_dict) 
		    sess.run(model.DE_train_op, feed_dict) 
		    		    
		    trg_pred_knn, logits_E_real,logits_E_fake,logits_G_real,logits_G_fake = sess.run([model.trg_pred_knn, model.logits_E_real,model.logits_E_fake,model.logits_G_real,model.logits_G_fake],feed_dict) 
		    
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
        source_images, source_labels = self.load_svhn(self.svhn_dir)

        with tf.Session(config=self.config) as sess:
	    
	    
            print ('Loading sampler generator.')
            variables_to_restore = slim.get_model_variables(scope='sampler_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)
	    
	    print ('Loading sampler discriminator.')
            variables_to_restore = slim.get_model_variables(scope='disc_e')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)
	    
            print ('Loading encoder.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
	    


	    # train model for source domain S
	    src_labels = utils.one_hot(source_labels[:1000],11)
	    src_noise = utils.sample_Z(1000,100,'uniform')

	    feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels}

	    samples = sess.run(model.sampled_images, feed_dict)

	    for i in range(1000):
		
		print str(i)+'/'+str(len(samples)), np.argmax(src_labels[i])
		plt.imshow(np.squeeze(samples[i]), cmap='gray')
		plt.imsave('./sample/'+str(i)+'_'+str(np.argmax(src_labels[i])),np.squeeze(samples[i]), cmap='gray')

    def check_TSNE(self):
	
	#~ usps_images, usps_labels = self.load_usps(self.usps_dir)
	source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	target_images, target_labels = self.load_usps(self.usps_dir)
	
	target_labels = np.squeeze(target_labels.T)
	target_labels-=1
	target_labels[target_labels==255] = 9
	
	
	#~ source_images = source_images[2000:4000]
	#~ source_labels = source_labels[2000:4000]
	#~ target_images = target_images[2000:3800]
	#~ target_labels = target_labels[2000:3800]
	

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
	    
	    while(True):
	    
		print ('Loading pretrained model.')
		variables_to_restore = slim.get_model_variables(scope='encoder')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.pretrained_model)
		
		
		print ('Loading sampler.')
		variables_to_restore = slim.get_model_variables(scope='sampler_generator')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.pretrained_sampler)
		
		
		summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
		saver = tf.train.Saver()

		n_samples =1000
       
		src_labels = utils.one_hot(source_labels[:n_samples],10)
		trg_labels = utils.one_hot(target_labels[:n_samples],10)
		src_noise = utils.sample_Z(n_samples,100,'uniform')
		
		feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.src_images: source_images[:n_samples], model.trg_images: target_images[:n_samples]}
		
		fzy, fx_src, fx_trg = sess.run([model.fzy, model.fx_src, model.fx_trg], feed_dict)
		
		src_labels = np.argmax(src_labels,1)
		trg_labels = np.argmax(trg_labels,1)

		print 'Computing T-SNE.'

		model_TSNE = TSNE(n_components=2, random_state=0)

		#~ TSNE_hA = model_TSNE.fit_transform(np.vstack((fzy,fx_src,fx_trg)))
		TSNE_hA = model_TSNE.fit_transform(np.vstack((fx_src,fzy)))
		       
	      
		plt.figure()
		#~ plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)), 3 * np.ones((n_samples,)))), s=3,  cmap = mpl.cm.jet)
		plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)))), s=3,  cmap = mpl.cm.jet)
		
		plt.figure()
		#~ plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels, src_labels, trg_labels, )), s=3,  cmap = mpl.cm.jet)
		plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels,src_labels)), s=3,  cmap = mpl.cm.jet)
			    
		plt.show()
		
		time.sleep(1)

    def test(self):
	
	# load svhn dataset
	src_images, src_labels = self.load_mnist(self.mnist_dir, split='train')
	src_test_images, src_test_labels = self.load_mnist(self.mnist_dir, split='test')

	trg_images, trg_labels = self.load_usps(self.usps_dir)
	trg_test_images, trg_test_labels = self.load_usps(self.usps_dir)
	
	trg_labels = np.squeeze(trg_labels.T)
	trg_test_labels = np.squeeze(trg_test_labels.T)
	trg_labels-=1
	trg_labels[trg_labels==255] = 9
	trg_test_labels-=1
	trg_test_labels[trg_labels==255] = 9
	
	#~ src_images = src_images[2000:4000]
	#~ src_labels = src_labels[2000:4000]
	#~ src_test_images = src_test_images[2000:4000]
	#~ src_test_labels = src_test_labels[2000:4000]
	
	#~ trg_images = trg_images[2000:3800]
	#~ trg_labels = trg_labels[2000:3800]
	#~ trg_test_images = trg_test_images[2000:3800]
	#~ trg_test_labels = trg_test_labels[2000:3800]

	# build a graph
	model = self.model
	model.build_model()
		
	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
	with tf.Session(config=self.config) as sess:
	    tf.global_variables_initializer().run()
	    saver = tf.train.Saver()
	    
	    t = 0
	    
	    while(True):
		
		print ('Loading model.')
		variables_to_restore = slim.get_model_variables(scope='encoder')
		restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.test_model)
		
		t+=1
    
		src_rand_idxs = np.random.permutation(src_test_images.shape[0])[:]
		trg_rand_idxs = np.random.permutation(trg_test_images.shape[0])[:]
		test_src_acc, test_trg_acc, _ = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.loss], 
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
	
		time.sleep(.20)
	    
if __name__=='__main__':

    from model import DSN
    model = DSN(mode='eval_dsn', learning_rate=0.0003)
    solver = Solver(model)
    solver.check_TSNE()











