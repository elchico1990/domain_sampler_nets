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
from sklearn.metrics import confusion_matrix

from scipy import misc

class Solver(object):

    def __init__(self, model, batch_size=60, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 svhn_dir='svhn', syn_dir='syn', mnist_dir='mnist', mnist_m_dir='mnist_m', usps_dir='usps', amazon_dir='amazon_reviews',
		 log_dir='logs', sample_save_path='sample', model_save_path='model', pretrained_model='model/model', gen_model='model/model_gen', pretrained_sampler='model/sampler', 
		 test_model='model/dtn', convdeconv_model = 'model/conv_deconv'):
        
        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        
	self.svhn_dir = '/data/'+svhn_dir
        self.mnist_dir = '/data/'+mnist_dir
        
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
	self.protocol = 'svhn_mnist' # possibilities: svhn_mnist, mnist_usps, mnist_usps_2, usps_mnist, syn_svhn, mnist_mnist_m, amazon_reviews

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
	
	
        return images[:2000], np.squeeze(labels).astype(int)[:2000]

    def pretrain(self):
	
	print 'Pretraining.'
        
	images, labels = self.load_mnist(self.mnist_dir, split='train')
	
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
		
		for start, end in zip(range(0, len(images), self.batch_size), range(self.batch_size, len(images), self.batch_size)):
		    
		    t+=1
		       
		    feed_dict = {model.images: images[start:end], model.labels: labels[start:end]} #trg here is just needed by the model but otherwise useless. 
		    
		    sess.run(model.train_op, feed_dict) 

		    if (t+1) % 250 == 0:
			summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
			rand_idxs = np.random.permutation(images.shape[0])[:1000]
			summary_writer.add_summary(summary, t)
			print 'Step: [%d/%d] loss: [%.6f] train acc: [%.2f]' %(t+1, self.pretrain_iter, l, acc)
			
		    if (t+1) % 250 == 0:
			saver.save(sess, os.path.join(self.model_save_path, 'model'))
	    
    def train_sampler(self):
	
	print 'Training sampler.'
        
	images, labels = self.load_mnist(self.mnist_dir, split='train')
	labels = utils.one_hot(labels, 10)
	
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
		
		for start, end in zip(range(0, len(images), batch_size), range(batch_size, len(images), batch_size)):
		    
		    t += 1

		    Z_samples = utils.sample_Z(batch_size, noise_dim, 'uniform')

		    feed_dict = {model.noise: Z_samples, model.images: images[start:end], model.labels: labels[start:end]}
	    
		    avg_D_fake = sess.run(model.logits_fake, feed_dict)
		    avg_D_real = sess.run(model.logits_real, feed_dict)
		    
		    sess.run(model.d_train_op, feed_dict)
		    sess.run(model.g_train_op, feed_dict)
		    
		    if (t+1) % 100 == 0:
			summary, dl, gl = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] d_loss: %.6f g_loss: %.6f avg_D_fake: %.2f avg_D_real: %.2f ' \
				   %(t+1, int(epochs*len(images) /batch_size), dl, gl, avg_D_fake.mean(), avg_D_real.mean()))
			
                    if (t+1) % 1000 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'sampler')) 

    def train_end_to_end(self):

	print 'Training sampler.'
        
	images, labels = self.load_mnist(self.mnist_dir, split='train')
	images_test, labels_test = self.load_mnist(self.mnist_dir, split='test')
	
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
            tf.global_variables_initializer().run()
            
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()
	     	    
	    t = 0
	    
	    print ('Loading pretrained model.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_model)
	    
	    print ('Loading sampler - generator.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    
	    print ('Loading sampler - discriminator.')
	    variables_to_restore = slim.get_model_variables(scope='disc_e')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
	    
	    balanced_labels = np.repeat(np.array([0,1,2,3,4,5,6,7,8,9]),6)
	    
	    for i in range(epochs):
		
		print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(images), batch_size), range(batch_size, len(images), batch_size)):
		    
		    t += 1

		    Z_samples = utils.sample_Z(batch_size, noise_dim, 'uniform')

		    feed_dict = {model.noise: Z_samples, model.balanced_labels: balanced_labels, model.images: images[start:end], model.labels: labels[start:end]}
	    
		    avg_D_fake = sess.run(model.logits_fake, feed_dict)
		    avg_D_real = sess.run(model.logits_real, feed_dict)
		    
		    #~ sess.run(model.train_op_images, feed_dict)		    
		    sess.run(model.train_op_features, feed_dict)
		    #~ sess.run(model.train_op_d, feed_dict)
		    #~ sess.run(model.train_op_g, feed_dict)
		
		    
		    if (t+1) % 100 == 0:
			summary, loss_d, loss_g, loss_images, loss_features = sess.run([model.summary_op, model.d_loss, model.g_loss, model.loss_images, model.loss_features], feed_dict)
			feed_dict[model.images] = images_test[:2000]
			feed_dict[model.labels] = labels_test[:2000]
			test_acc = sess.run([model.accuracy_images], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] img_loss: %.3f feat_loss: %.3f d_loss: %.3f g_loss: %.3f avg_D_fake: %.2f avg_D_real: %.2f accuracy: %.6f ' \
				   %(t+1, int(epochs*len(images) /batch_size), loss_images, loss_features, loss_d, loss_g, avg_D_fake.mean(), avg_D_real.mean(), test_acc[0]))
			
                    if (t+1) % 1000 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'end_to_end')) 

    def check_TSNE(self):
	
	images, labels = self.load_mnist(self.mnist_dir, split='train')
	
        # build a graph
        model = self.model
        model.build_model()

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
	    
	    else:
		raise NameError('Unrecognized mode.')
	    
            
	    n_samples = 2000
            labels = utils.one_hot(labels[:n_samples],10)
	    noise = utils.sample_Z(n_samples,100,'uniform')
	   
		
	    print ('Loading sampler.')
	    variables_to_restore = slim.get_model_variables(scope='sampler_generator')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.pretrained_sampler)
    
	    
	    feed_dict = {model.noise: noise, model.labels: labels, model.images: images[:n_samples]}
	    
	    fzy, fx = sess.run([model.fzy, model.fx], feed_dict)
	    
	    labels = np.argmax(labels,1)
	    
	    print 'Computing T-SNE.'

	    model = TSNE(n_components=2, random_state=0)

	       
	    if sys.argv[2] == '1':
		TSNE_hA = model.fit_transform(fx)
		f, (ax1,ax2) = plt.subplots(1,2,sharey=True)
		ax1.set_facecolor('white')
		ax2.set_facecolor('white')
		ax1.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.ones((n_samples)), s=3, cmap = mpl.cm.jet, alpha=0.5)
		ax2.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = labels, s=3,  cmap = mpl.cm.jet, alpha=0.5)
		
	    elif sys.argv[2] == '2':
		TSNE_hA = model.fit_transform(np.vstack((fzy,fx)))
		f, (ax1,ax2) = plt.subplots(1,2,sharey=True)
		ax1.set_facecolor('white')
		ax2.set_facecolor('white')
		ax1.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)))), s=3, cmap = mpl.cm.jet, alpha=0.5)
		ax2.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((labels,labels)), s=3, cmap = mpl.cm.jet, alpha=0.5)


	    plt.legend()
	    plt.show()
	    
	    
	    
	    
	    
	    
	    
	    


	    
	    
	    
	    
	    
	    
    def test(self):
	
	# build a graph
	model = self.model
	model.build_model()
	
        
	test_images, test_labels = self.load_mnist(self.mnist_dir, split='test')
	
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
    
		test_src_acc, test_trg_acc, trg_pred = sess.run(fetches=[model.src_accuracy, model.trg_accuracy, model.trg_pred], 
				       feed_dict={model.src_images: test_images, 
						  model.src_labels: test_labels,
						  model.trg_images: test_images, 
						  model.trg_labels: test_labels})
						  
		print ('Step: [%d/%d] test acc [%.3f]' \
			   %(t+1, self.pretrain_iter, test_trg_acc))
		
		print confusion_matrix(test_labels, trg_pred)	   
		
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
    model = DSN(mode='eval_dsn', learning_rate=0.0003)
    solver = Solver(model)
    #~ solver.find_closest_samples()
    
    solver.check_TSNE()

