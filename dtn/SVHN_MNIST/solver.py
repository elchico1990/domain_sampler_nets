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
import zmq
import matplotlib.pyplot as plt
import matplotlib as mpl
#~ import seaborn as sns

from sklearn.metrics import confusion_matrix

import utils
#~ from sklearn.manifold import TSNE

from scipy import misc


class Solver(object):

    def __init__(self, model, batch_size=64, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 svhn_dir='../data/svhn', mnist_dir='../data/mnist', usps_dir='usps', log_dir='logs', sample_save_path='sample', 
                 model_save_path='', pretrained_model='model', pretrained_sampler='sampler', 
		 test_model='dtn', convdeconv_model = 'conv_deconv'):
        
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
        
	if sys.argv[1] == 'adda':
	    self.model_save_path='/cvgl2/u/rvolpi/SVHN_MNIST/model/adda'
	    
	elif sys.argv[1] == 'adda_di':
	    self.model_save_path='/cvgl2/u/rvolpi/SVHN_MNIST/model/adda_di'
	    
	elif sys.argv[1] == 'fa':
	    self.model_save_path='/cvgl2/u/rvolpi/SVHN_MNIST/model/fa'
	    
	else:
	    self.model_save_path='/cvgl2/u/rvolpi/SVHN_MNIST/model'
	    
	self.pretrained_model = os.path.join(self.model_save_path,pretrained_model)
	self.pretrained_sampler = os.path.join(self.model_save_path,pretrained_sampler)
	self.test_model = os.path.join(self.model_save_path,test_model)
	
	
	self.convdeconv_model = convdeconv_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	self.protocol = 'svhn_mnist'

    def load_svhn(self, image_dir, split='train'):
        print ('Loading SVHN dataset.')
        
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

    def load_gen_images(self):
	
	'''
	Loading images generated with eval_dsn()
	Assuming that image_dir contains folder with
	subfolders 1,2,...,9.
	'''
	
	print 'Loading generated images.'
	
	no_images = 10000 # number of images per digit
	
	labels = np.zeros((10 * no_images,)).astype(int)
	images = np.zeros((10 * no_images,32,32,3))
	
	for l in range(10):
	    print l
	    counter = 0
	    for img_dir in sorted(glob.glob('/home/rvolpi/Desktop/domain_sampler_nets/dtn/sample/'+str(l)+'/*'))[:no_images]:
		im = misc.imread(img_dir)
		im = np.expand_dims(im[:,:,:3], axis=0)
		images[l * no_images + counter] = im
		labels[l * no_images + counter] = l
		counter+=1
	    
	print 'break'
	images = images / 127.5 - 1
	return images, labels
	
    def pretrain(self):
        # load svhn dataset
        src_images, src_labels = self.load_svhn(self.svhn_dir, split='train')
        src_test_images, src_test_labels = self.load_svhn(self.svhn_dir, split='test')

        trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train')
        trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')

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

	    epochs = 30
	    
	    t = 0

	    for i in range(epochs):
		
		print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(src_images), self.batch_size), range(self.batch_size, len(src_images), self.batch_size)):
		    
		    t+=1
		       
		    feed_dict = {model.src_images: src_images[start:end], model.src_labels: src_labels[start:end], model.trg_images: trg_images[0:2], model.trg_labels: trg_labels[0:2]} #trg here is just needed by the model but otherwise useless. 
		    
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
        # load svhn dataset
        source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	source_labels = utils.one_hot(source_labels, 10)
	
	#~ svhn_images = svhn_images[np.where(np.argmax(svhn_labels,1)==1)]
	#~ svhn_labels = svhn_labels[np.where(np.argmax(svhn_labels,1)==1)]
        
        # build a graph
        model = self.model
        model.build_model()
#~ 
        #~ # make directory if not exists
        #~ if tf.gfile.Exists(self.log_dir):
            #~ tf.gfile.DeleteRecursively(self.log_dir)
        #~ tf.gfile.MakeDirs(self.log_dir)
	#~ 
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
        
	print 'Training DSN.'
	
	if self.protocol=='svhn_mnist':
	    source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	    target_images, target_labels = self.load_mnist(self.mnist_dir, split='train')


	algorithm = str(sys.argv[1])
	
        # build a graph
        model = self.model
        model.build_model(algorithm)

	
	source_features = np.zeros((len(source_images),128))
	
	#~ self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
	with tf.Session(config=self.config) as sess:
	    	    
	    tf.global_variables_initializer().run()
	    
	    print ('Loading pretrained encoder.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_model)
	    
	    print 'Extracting source features'
	    
	    for indices, source_images_batch in zip(np.array_split(np.arange(len(source_images)),20), np.array_split(source_images,20)):
		print indices[0]
		source_features[indices] = sess.run(model.orig_src_fx, feed_dict={model.src_features: np.zeros((1,128)), model.src_images: source_images_batch, model.src_noise: np.zeros((1,100)), model.src_labels: utils.one_hot(source_labels,10), model.trg_images: target_images[0:1]})

        
	    	
	tf.reset_default_graph()

	# build a graph
        model = self.model
        model.build_model(algorithm)


	with tf.Session(config=self.config) as sess:
	    	    
	    # initialize G and D
	    tf.global_variables_initializer().run()
	    # restore variables of F
	    
	    print ('Loading pretrained encoder.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_model)
	    
	    #~ print ('Loading pretrained discriminator.')
	    #~ variables_to_restore = slim.get_model_variables(scope='disc_e')
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

	    context = zmq.Context()
	    socket = context.socket(zmq.DEALER)
	    
	    if sys.argv[1] == 'adda':
		socket.connect('tcp://localhost:5560')
	    elif sys.argv[1] == 'adda_di':
		socket.connect('tcp://localhost:5660')
	    if sys.argv[1] == 'fa':
		socket.connect('tcp://localhost:5760')
	    

	    for step in range(40001):
		
		trg_count += 1
		t+=1
		
		i = step % int(source_images.shape[0] / self.batch_size)
		j = step % int(target_images.shape[0] / self.batch_size)
		
		src_images = source_images[i*self.batch_size:(i+1)*self.batch_size]
		src_labels = utils.one_hot(source_labels[i*self.batch_size:(i+1)*self.batch_size],10)
		src_labels_int = source_labels[i*self.batch_size:(i+1)*self.batch_size]
		src_noise = utils.sample_Z(self.batch_size,100,'uniform')
		trg_images = target_images[j*self.batch_size:(j+1)*self.batch_size]
		src_features = source_features[i*self.batch_size:(i+1)*self.batch_size]
		
		feed_dict = {model.src_features: src_features, model.src_images: src_images, model.src_noise: src_noise, model.src_labels: src_labels, model.trg_images: trg_images}
		
		
		sess.run(model.E_train_op, feed_dict) 
		sess.run(model.DE_train_op, feed_dict)

		logits_E_real,logits_E_fake = sess.run([model.logits_E_real,model.logits_E_fake],feed_dict) 
		
		if (step) % 1000 == 0:
		    
		    summary, E, DE = sess.run([model.summary_op, model.E_loss, model.DE_loss], feed_dict)
		    summary_writer.add_summary(summary, step)
		    print ('Step: [%d/%d] E: [%.6f] DE: [%.6f] E_real: [%.2f] E_fake: [%.2f]' \
			       %(step+1, self.train_iter, E, DE, logits_E_real.mean(),logits_E_fake.mean()))

		    

		if (step) % 1000 == 0:
		    print 'Saving...'
		    saver.save(sess, self.test_model)
		    print 'Sending...'
		    socket.send_string(algorithm)
		    
    def eval_dsn(self):
        # build model
        model = self.model
        model.build_model()

	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
        with tf.Session(config=self.config) as sess:
	    
	    #~ print ('Loading pretrained G.')
	    #~ variables_to_restore = slim.get_model_variables(scope='generator')
	    #~ restorer = tf.train.Saver(variables_to_restore)
	    #~ restorer.restore(sess, self.test_model)
	    
	    print ('Loading pretrained E.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_model)
	    
	    print ('Loading sample generator.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    
	    #~ source_images, source_labels = self.load_mnist(self.mnist_dir, split='train')
	    #~ source_images = source_images[:2000]
	    #~ source_labels = source_labels[:2000]
	    
	    source_images, source_labels = self.load_usps(self.usps_dir)
	    source_images = source_images[:1800]
	    source_labels = source_labels[:1800]
	    
	    source_images = np.repeat(source_images,50,0)
	    source_labels = np.repeat(source_labels,50,0)

	    #~ for n in range(0,10):
		
		#~ print n
	    
		#~ no_gen = 10000

		#~ source_labels = n * np.ones((no_gen,),dtype=int)

		#~ # train model for source domain S
		#~ src_labels = utils.one_hot(source_labels[:no_gen],10)
		#~ src_noise = utils.sample_Z(no_gen,100,'uniform')

		#~ feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels}

		#~ samples, samples_logits = sess.run([model.sampled_images, model.sampled_images_logits], feed_dict)
		#~ samples_logits = samples_logits[:,n]
		#~ samples = samples[samples_logits>8.]
		#~ samples_logits = samples_logits[samples_logits>8.]
		
		#~ for i in range(len(samples_logits)):
		    
		    ## print str(i)+'/'+str(len(samples_logits))-
		    
		    #~ plt.imshow(np.squeeze(samples[i]), cmap='gray')
		    #~ plt.imsave('./sample/'+str(np.argmax(src_labels[i]))+'/'+str(i)+'_'+str(np.argmax(src_labels[i]))+'_'+str(samples_logits[i]),np.squeeze(samples[i]), cmap='gray')
		
		#~ print str(i)+'/'+str(len(samples)), np.argmax(src_labels[i])

	    no_gen = len(source_images)
	    
	    print 'Number of samples:',no_gen

	    # train model for source domain S
	    src_images = source_images[:2]
	    src_labels = utils.one_hot(source_labels[:no_gen],10)
	    src_noise = utils.sample_Z(no_gen,100,'uniform')

	    feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.src_images: src_images}

	    fzy, fx_src, fzy_labels = sess.run([model.fzy, model.fx_src, model.fzy_labels], feed_dict)
	    
	    fzy_states = (fzy>0.).astype(int)
	    fx_src_states = (fx_src>0.).astype(int)
	    
	    tmpUnique = np.unique(fzy_states.view(np.dtype((np.void, fzy_states.dtype.itemsize*fzy_states.shape[1]))), return_counts = True)
	    fzy_states_unique = tmpUnique[0].view(fzy_states.dtype).reshape(-1, fzy_states.shape[1])
	    print 'fzy:',fzy_states_unique.shape
	    
	    tmpUnique = np.unique(fx_src_states.view(np.dtype((np.void, fx_src_states.dtype.itemsize*fx_src_states.shape[1]))), return_counts = True)
	    fx_src_states_unique = tmpUnique[0].view(fx_src_states.dtype).reshape(-1, fx_src_states.shape[1]) 
	    print 'fx_src:',fx_src_states_unique.shape
	    
	    print (np.argmax(src_labels,1)==fzy_labels).astype(int).mean()
	    
	    while(True):
		
		src_images = source_images[:2]
		src_labels = utils.one_hot(source_labels[:no_gen],10)
		src_noise = utils.sample_Z(no_gen,100,'uniform')

		feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.src_images: src_images}

		fzy, fx_src = sess.run([model.fzy, model.fx_src], feed_dict)
		
		fzy_states = (fzy>0.).astype(int)
		fx_src_states = (fx_src>0.).astype(int)
		
		fzy_states = np.vstack((fzy_states, fzy_states_unique))
		fx_src_states = np.vstack((fx_src_states, fx_src_states_unique))
		
		tmpUnique = np.unique(fzy_states.view(np.dtype((np.void, fzy_states.dtype.itemsize*fzy_states.shape[1]))), return_counts = True)
		fzy_states_unique = tmpUnique[0].view(fzy_states.dtype).reshape(-1, fzy_states.shape[1])
		print 'fzy:',fzy_states_unique.shape
		
		    
		
	    
	    print 'break'
		    
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
			trg_rand_idxs = np.random.permutation(trg_images.shape[0])[:]
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
	    
	elif self.protocol == 'usps_mnist':
	    source_images, source_labels = self.load_usps(self.usps_dir)
	    target_images, target_labels = self.load_mnist(self.mnist_dir)
	
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
	
	
	context = zmq.Context()
	socket = context.socket(zmq.DEALER)
	
	if sys.argv[1] == 'adda':
	    socket.connect('tcp://localhost:5570')
	elif sys.argv[1] == 'adda_di':
	    socket.connect('tcp://localhost:5670')
	elif sys.argv[1] == 'fa':
	    socket.connect('tcp://localhost:5770')
	
	if self.protocol == 'svhn_mnist':
	    
	    src_images, src_labels = self.load_svhn(self.svhn_dir, split='train')
	    src_test_images, src_test_labels = self.load_svhn(self.svhn_dir, split='test')
	    
	    trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train')
	    trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')
	

	
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
		
		print 'Waiting...'
		
		algorithm = socket.recv_string()
		
		if sys.argv[2] == 'test':
		    print ('Loading test model.')
		    variables_to_restore = slim.get_model_variables(scope='encoder')
		    restorer = tf.train.Saver(variables_to_restore)
		    restorer.restore(sess, self.test_model)
		
		elif sys.argv[2] == 'pretrain':
		    print ('Loading pretrained model.')
		    variables_to_restore = slim.get_model_variables(scope='encoder')
		    restorer = tf.train.Saver(variables_to_restore)
		    restorer.restore(sess, self.pretrained_model)
		
		elif sys.argv[2] == 'gen':
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
		src_acc = sess.run(model.src_accuracy, feed_dict={model.src_images: src_images[:20000], 
								  model.src_labels: src_labels[:20000],
						                  model.trg_images: trg_test_images[trg_rand_idxs], 
								  model.trg_labels: trg_test_labels[trg_rand_idxs]})
						  
		print ('Step: [%d/%d] src train acc [%.3f]  src test acc [%.3f] trg test acc [%.3f]' \
			   %(t+1, self.pretrain_iter, src_acc, test_src_acc, test_trg_acc))
		
		print confusion_matrix(trg_test_labels[trg_rand_idxs], trg_pred)	   
		
		acc.append(test_trg_acc)
		with open(self.protocol + '_' + algorithm + '.pkl', 'wb') as f:
		    cPickle.dump(acc,f,cPickle.HIGHEST_PROTOCOL)
      
if __name__=='__main__':

    from model import DSN
    model = DSN(mode='eval_dsn', learning_rate=0.0003)
    solver = Solver(model)
    solver.check_TSNE()
