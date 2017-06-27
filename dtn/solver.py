import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
import cPickle

import matplotlib.pyplot as plt

import utils

from sklearn.manifold import TSNE

class Solver(object):

    def __init__(self, model, batch_size=64, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 svhn_dir='svhn', mnist_dir='mnist', usps_dir='usps', log_dir='logs', sample_save_path='sample', 
                 model_save_path='model', pretrained_model='model/model', pretrained_sampler='model/sampler', 
		 test_model='model/dtn', adda_model='model/adda', pretrained_adda_model='model/pre_adda'):
        
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
        self.adda_model = adda_model
        self.pretrained_adda_model = pretrained_adda_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=False

    def load_svhn(self, image_dir, split='train'):
        print ('Loading SVHN dataset.')
        
        if self.model.mode in ['pretrain','adda_pretrain']:
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
        source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
        test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
        
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    epochs = 200
	    
	    t = 0

	    for i in range(epochs):
		
		print 'Epoch',str(i)
		print len(source_images)
		for start, end in zip(range(0, len(source_images), self.batch_size), range(self.batch_size, len(source_images), self.batch_size)):
		    
		    t+=1
		       
		    batch_images = source_images[start:end]
		    batch_labels = source_labels[start:end] 
		    feed_dict = {model.images: batch_images, model.labels: batch_labels}
		    
		    a,b = sess.run([model.labels, model.logits], feed_dict)
		    
		    sess.run(model.train_op, feed_dict) 

		    if (t+1) % 500 == 0:
			summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
			rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
			test_acc, _ = sess.run(fetches=[model.accuracy, model.loss], 
					       feed_dict={model.images: test_images[rand_idxs], 
							  model.labels: test_labels[rand_idxs]})
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
				   %(t+1, self.pretrain_iter, l, acc, test_acc))

		    if (t+1) % 1000 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'model')) 
		    
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

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	
	batch_size = self.batch_size
	noise_dim = 100
	epochs = 300

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
            print ('Loading pretrained model.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            # restore variables of F
	    
            print ('Loading sampler.')
            variables_to_restore = slim.get_model_variables(scope='sampler_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)
	    
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
        
	target_images, target_labels = self.load_mnist(self.mnist_dir, split='train')
	#~ usps_images, usps_labels = self.load_usps(self.usps_dir)
	source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	

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
	    
            print ('Loading sample generator.')
            variables_to_restore = slim.get_model_variables(scope='sampler_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)
	    

	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('Start training.')
            trg_count = 0
            for step in range(self.train_iter+1):
		
		trg_count += 1
                
		i = step % int(source_images.shape[0] / self.batch_size)
                j = step % int(target_images.shape[0] / self.batch_size)
                
		src_images = source_images[i*self.batch_size:(i+1)*self.batch_size]
                src_labels = utils.one_hot(source_labels[i*self.batch_size:(i+1)*self.batch_size],10)
		src_labels_int = source_labels[i*self.batch_size:(i+1)*self.batch_size]
		src_noise = utils.sample_Z(self.batch_size,100,'uniform')
		trg_images = target_images[j*self.batch_size:(j+1)*self.batch_size]
		
		feed_dict = {model.src_images: src_images, model.src_noise: src_noise, model.src_labels: src_labels, model.trg_images: trg_images}
		
		sess.run(model.E_train_op, feed_dict) 
		
		sess.run(model.DE_train_op, feed_dict) 
		
		sess.run(model.G_train_op, feed_dict) 
		
		sess.run(model.DG_train_op, feed_dict) 
		
		sess.run(model.const_train_op, feed_dict) 
		
                if (step+1) % 10 == 0:
		    
		    summary, E, DE, G, DG, cnst = sess.run([model.summary_op, model.E_loss, model.DE_loss, model.G_loss, model.DG_loss, model.const_loss], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] D: [%.6f] DE: [%.6f] G: [%.6f] DG: [%.6f] Const: [%.6f]' \
                               %(step+1, self.train_iter, E, DE, G, DG, cnst))

                if (step+1) % 500 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'))
            
    def eval_dsn(self):
        # build model
        model = self.model
        model.build_model()

        # load svhn dataset
        source_images, source_labels = self.load_svhn(self.svhn_dir)

        with tf.Session(config=self.config) as sess:
	    
	    
            print ('Loading sampler.')
            variables_to_restore = slim.get_model_variables(scope='sampler_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)
	    
            print ('Loading generator.')
            variables_to_restore = slim.get_model_variables(scope='generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.test_model)
	    


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
	
	target_images, target_labels = self.load_mnist(self.mnist_dir, split='train')
	#~ usps_images, usps_labels = self.load_usps(self.usps_dir)
	source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	

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
	    
            print ('Loading pretrained model.')
            variables_to_restore = slim.get_model_variables(scope='encoder')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
	    
            print ('Loading sampler.')
            variables_to_restore = slim.get_model_variables(scope='sampler_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)
	    
            #~ print ('Loading test model.')
            #~ saver = tf.train.Saver()
            #~ saver.restore(sess, self.test_model)

	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

   
	    src_labels = utils.one_hot(source_labels[:1000],10)
	    trg_labels = utils.one_hot(target_labels[:1000],10)
	    src_noise = utils.sample_Z(1000,100,'uniform')
	    
	    feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.src_images: source_images[:1000], model.trg_images: target_images[:1000]}
	    
	    src_fx, trg_fx, fx = sess.run([model.orig_src_fx, model.orig_trg_fx, model.fzy], feed_dict)
	    
	    src_labels = np.argmax(src_labels,1)
	    trg_labels = np.argmax(trg_labels,1)

	    print 'Computing T-SNE.'

	    model = TSNE(n_components=2, random_state=0)

	    print '4'
	    TSNE_hA_4 = model.fit_transform(np.vstack((src_fx,fx)))

	    #~ print '5'
	    #~ TSNE_hA_5 = model.fit_transform(np.vstack((src_fx,fgfx,trg_fx)))

	    #~ print '6'
	    #~ TSNE_hA_6 = model.fit_transform(np.vstack((fx,fgfx)))
		   
	    plt.figure(6)
	    plt.scatter(TSNE_hA_4[:,0], TSNE_hA_4[:,1], c = np.hstack((np.ones((1000,)), 2 * np.ones((1000,)))))
	    
	    plt.figure(7)
	    plt.scatter(TSNE_hA_4[:,0], TSNE_hA_4[:,1], c = np.hstack((src_labels,src_labels)))
		    
	    #~ plt.figure(8)
	    #~ plt.scatter(TSNE_hA_5[:,0], TSNE_hA_5[:,1], c = np.hstack((np.ones((500,)), 2 * np.ones((500,)), 3 * np.ones((500,)))))
	    
	    #~ plt.figure(9)
	    #~ plt.scatter(TSNE_hA_5[:,0], TSNE_hA_5[:,1], c = np.hstack((src_labels,src_labels,trg_labels)))
		  
	    #~ plt.figure(10)
	    #~ plt.scatter(TSNE_hA_6[:,0], TSNE_hA_6[:,1], c = np.hstack((np.ones((500,)), 2 * np.ones((500,)))))
	    
	    #~ plt.figure(11)
	    #~ plt.scatter(TSNE_hA_6[:,0], TSNE_hA_6[:,1], c = np.hstack((src_labels,src_labels)))
		       
	    plt.show()

if __name__=='__main__':

    from model import DSN
    model = DSN(mode='train_dsn', learning_rate=0.0003)
    solver = Solver(model)
    solver.check_TSNE()







