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

class Solver(object):

    def __init__(self, model, batch_size=64, pretrain_iter=100000, train_iter=10000, sample_iter=2000, 
                 svhn_dir='svhn', mnist_dir='mnist', usps_dir='usps', log_dir='logs', sample_save_path='sample', 
                 model_save_path='model', pretrained_model='model/model-100000', pretrained_sampler='model/sampler-45000', 
		 test_model='model/dtn-1000', adda_model='model/adda-1500'):
        
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
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=False

    def load_svhn(self, image_dir, split='train'):
        print ('Loading SVHN dataset.')
        
        if self.model.mode == 'pretrain':
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
        train_images, train_labels = self.load_svhn(self.svhn_dir, split='train')
        test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
        
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter+1):
                i = step % int(train_images.shape[0] / self.batch_size)
                batch_images = train_images[i*self.batch_size:(i+1)*self.batch_size]
                batch_labels = train_labels[i*self.batch_size:(i+1)*self.batch_size] 
                feed_dict = {model.images: batch_images, model.labels: batch_labels}
                
		a,b = sess.run([model.labels, model.logits], feed_dict)
		
		sess.run(model.train_op, feed_dict) 

                if (step+1) % 500 == 0:
                    summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
                    rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc, _ = sess.run(fetches=[model.accuracy, model.loss], 
                                           feed_dict={model.images: test_images[rand_idxs], 
                                                      model.labels: test_labels[rand_idxs]})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                               %(step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:  
                    saver.save(sess, os.path.join(self.model_save_path, 'model'), global_step=step+1) 
                    print ('model-%d saved..!' %(step+1))
	
    def adda_train(self):
	
	print 'Training ADDA encoder.'
        # load svhn dataset
        source_images, _ = self.load_svhn(self.svhn_dir, split='train')
        target_images, _ = self.load_mnist(self.mnist_dir, split='train')
	
        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	
	batch_size = 64
	noise_dim = 100
	epochs = 300

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
            print ('Loading pretrained model.')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            # restore variables of F
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()
		 	    
	    t = 0
    
	    for i in range(epochs):
		
		#~ print 'Epoch',str(i)
		
		for start, end in zip(range(0, min(len(source_images),len(target_images)), batch_size), range(batch_size, min(len(source_images),len(target_images)), batch_size)):
		    
		    t += 1

		    feed_dict = {model.src_images: source_images[start:end], model.trg_images: target_images[start:end]}

		    sess.run(model.d_train_op, feed_dict)
		    sess.run(model.enc_train_op, feed_dict)
		    
		    avg_D_trg = sess.run(model.logit_trg, feed_dict)
		    avg_D_src = sess.run(model.logit_src, feed_dict)
		    
		    		    
		    if (t+1) % 100 == 0:
			summary, dl, encl = sess.run([model.summary_op, model.d_loss, model.enc_loss], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] d_loss: [%.6f] enc_loss: [%.6f]' \
				   %(t+1, int(epochs*len(source_images) /batch_size), dl, encl))
			print 'avg_D_trg',str(avg_D_trg.mean()),'avg_D_src',str(avg_D_src.mean())
			
                    if (t+1) % 500 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'adda'), global_step=t+1) 
		    
    def train_sampler(self):
	
	print 'Training sampler.'
        # load svhn dataset
        source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
	source_labels_oh = utils.one_hot(source_labels, 11)
	
	#~ svhn_images = svhn_images[np.where(np.argmax(svhn_labels,1)==1)]
	#~ svhn_labels = svhn_labels[np.where(np.argmax(svhn_labels,1)==1)]
        
        # build a graph
        model = self.model
        model.build_model()

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	
	batch_size = 64
	noise_dim = 100
	epochs = 300

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
            # restore variables of F
            print ('Loading pretrained model.')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            # restore variables of F
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()
	    
	    #~ feed_dict = {model.images: source_images[:10000]}
	    #~ fx = sess.run(model.fx, feed_dict)
		 	    
	    t = 0
	    
	    labels_fake_oh = np.zeros((batch_size,11))
	    labels_fake_oh[:,10] = 1
	    labels_fake = np.argmax(labels_fake_oh,1)
	    
	    for i in range(epochs):
		
		#~ print 'Epoch',str(i)
		
		for start, end in zip(range(0, len(source_images), batch_size), range(batch_size, len(source_images), batch_size)):
		    
		    t += 1

		    Z_samples = utils.sample_Z(batch_size, noise_dim, 'uniform')

		    feed_dict = {model.noise: Z_samples, model.images: source_images[start:end], model.labels_real: source_labels[start:end], model.labels_fake: labels_fake, model.labels_real_oh: source_labels_oh[start:end], model.labels_fake_oh: labels_fake_oh}

		    #~ a,b,c,d = sess.run([model.logits_real,model.logits_fake,model.labels_real,model.labels_fake], feed_dict)
		    		    
		    #~ avg_D_fake = sess.run(model.logits_fake, feed_dict)
		    #~ avg_D_real = sess.run(model.logits_real, feed_dict)
		    
		    sess.run(model.d_train_op, feed_dict)
		    sess.run(model.g_train_op, feed_dict)
		    
		    if (t+1) % 100 == 0:
			summary, dl, gl = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
			summary_writer.add_summary(summary, t)
			print ('Step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
				   %(t+1, int(epochs*len(source_images) /batch_size), dl, gl))
			#~ print 'avg_D_fake',str(avg_D_fake.mean()),'avg_D_real',str(avg_D_real.mean())
			
                    if (t+1) % 1000 == 0:  
			saver.save(sess, os.path.join(self.model_save_path, 'sampler'), global_step=t+1) 
			print ('sampler-%d saved..!' %(t+1))

    def train(self):
        # load svhn dataset
        svhn_images, _ = self.load_svhn(self.svhn_dir, split='train')
        mnist_images, _ = self.load_mnist(self.mnist_dir, split='train')

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
            print ('loading pretrained model F..')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('start training..!')
            f_interval = 15
            for step in range(self.train_iter+1):
                
                i = step % int(svhn_images.shape[0] / self.batch_size)
                # train the model for source domain S
                src_images = svhn_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.src_images: src_images}
                
                sess.run(model.d_train_op_src, feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict) 
                sess.run([model.g_train_op_src], feed_dict)
                
                if step > 1600:
                    f_interval = 30
                
                if i % f_interval == 0:
                    sess.run(model.f_train_op_src, feed_dict)
                
                if (step+1) % 10 == 0:
                    summary, dl, gl, fl = sess.run([model.summary_op_src, \
                        model.d_loss_src, model.g_loss_src, model.f_loss_src], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl, fl))
                
                # train the model for target domain T
                j = step % int(mnist_images.shape[0] / self.batch_size)
                trg_images = mnist_images[j*self.batch_size:(j+1)*self.batch_size]
                feed_dict = {model.src_images: src_images, model.trg_images: trg_images}
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.d_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)
                sess.run(model.g_train_op_trg, feed_dict)

                if (step+1) % 10 == 0:
                    summary, dl, gl = sess.run([model.summary_op_trg, \
                        model.d_loss_trg, model.g_loss_trg], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl))

                if (step+1) % 200 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
                    print ('model/dtn-%d saved' %(step+1))
    
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
            #~ print ('Loading ADDA encoder.')
            #~ variables_to_restore = slim.get_model_variables(scope='target_encoder')
            #~ restorer = tf.train.Saver(variables_to_restore)
            #~ restorer.restore(sess, self.adda_model)
            
            print ('Loading pretrained model F.')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            
            print ('Loading sampler.')
            variables_to_restore = slim.get_model_variables(scope='sampler_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_sampler)

	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('Start training.')
            trg_count = 0
            for step in range(self.train_iter+1):
		
		trg_count += 1
                
                
		#~ src_labels = utils.one_hot(source_labels[:500],11)
		#~ trg_labels = utils.one_hot(target_labels[:500],11)
		#~ src_noise = utils.sample_Z(500,100)
		
		#~ feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.src_images: source_images[:500], model.trg_images: target_images[:500]}
		
		#~ src_fx, trg_fx, adda_trg_fx, fx = sess.run([model.orig_src_fx, model.orig_trg_fx, model.adda_trg_fx, model.fx], feed_dict)
		
		#~ f = file('./for_tsne.pkl','w')
		#~ cPickle.dump((fx, src_fx, src_labels, trg_fx, adda_trg_fx, trg_labels),f,cPickle.HIGHEST_PROTOCOL) 
		#~ f.close()
		
		i = step % int(source_images.shape[0] / self.batch_size)
                j = step % int(target_images.shape[0] / self.batch_size)
                
		src_images = source_images[i*self.batch_size:(i+1)*self.batch_size]
                src_labels = utils.one_hot(source_labels[i*self.batch_size:(i+1)*self.batch_size],11)
		src_labels_int = source_labels[i*self.batch_size:(i+1)*self.batch_size]
		src_noise = utils.sample_Z(self.batch_size,100)
		noise = utils.sample_Z(self.batch_size,100)
		trg_images = target_images[j*self.batch_size:(j+1)*self.batch_size]
                		
		feed_dict = {model.src_images: src_images, model.src_noise: src_noise, model.src_labels: src_labels, model.src_labels_int: src_labels_int, model.trg_images: trg_images, model.noise: noise}
		
		# Training D to classify well images generated from SRC
		sess.run(model.d_train_op_src, feed_dict) 
		
		# Training G to fool D in classifying images generated from RSC
		sess.run(model.g_train_op_src, feed_dict) 
		
		# Forcing hidden representation of images generated from SRC to 
	        # be close to hidden representation used as starting point
		sess.run(model.f_train_op_src, feed_dict) # FORCING LABELS NOW
		
		# Training D to classufy well images generated from TRG
		sess.run(model.d_train_op_trg, feed_dict)
                
		# Training G to fool D in classifying images generated from TRG
		sess.run(model.g_train_op_trg, feed_dict)
		
		# Forcing images generated from TRG to be close to images
		# used as starting point
		sess.run(model.g_train_op_const_trg, feed_dict)
		
		
                if (step+1) % 10 == 0:
		    
		    summary, dl, gl, fl = sess.run([model.summary_op_src, \
                        model.d_loss_src, model.g_loss_src, model.f_loss_src], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] f_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl, fl))
                
                    summary, dl, gl, cl = sess.run([model.summary_op_trg, \
                        model.d_loss_trg, model.g_loss_trg, model.g_loss_const_trg], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Target] step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f] const_loss: [%.6f]' \
                               %(step+1, self.train_iter, dl, gl, cl))



                if (step+1) % 50 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step+1)
        
    def eval(self):
        # build model
        model = self.model
        model.build_model()

        # load svhn dataset
        svhn_images, _ = self.load_svhn(self.svhn_dir)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = svhn_images[i*self.batch_size:(i+1)*self.batch_size]
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
	    
    def eval_dsn(self):
        # build model
        model = self.model
        model.build_model()

        # load svhn dataset
        mnist_images, mnist_labels = self.load_mnist(self.mnist_dir)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)


	    # train model for source domain S
	    src_labels = utils.one_hot(mnist_labels[:1000],11)
	    src_noise = utils.sample_Z(1000,100)
	    noise = utils.sample_Z(1000,100)

	    feed_dict = {model.src_noise: src_noise, model.src_labels: src_labels, model.noise: noise}

	    samples = sess.run(model.sampled_images, feed_dict)

	    for i in range(1000):
		
		print str(i)+'/'+str(len(samples)), np.argmax(src_labels[i])
		plt.imshow(np.squeeze(samples[i]), cmap='gray')
		plt.imsave('./sample/'+str(i)+'_'+str(np.argmax(src_labels[i])),np.squeeze(samples[i]), cmap='gray')

	    path = os.path.join(self.sample_save_path, 'sample-%d-to-%d.png' %(i*self.batch_size, (i+1)*self.batch_size))
	    scipy.misc.imsave(path,sampled_batch_images)
	    print ('saved %s' %path)


if __name__=='__main__':
    
    uspsData = scipy.io.loadmat('./usps/USPS.mat')
    
    print 'break'
























		    
