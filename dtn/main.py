import tensorflow as tf
from model import DSN
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('model_save_path', 'model', "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample', "directory for saving the sampled images")
FLAGS = flags.FLAGS

def main(_):
    
    model = DSN(mode=FLAGS.mode, learning_rate=0.0003)
    solver = Solver(model, svhn_dir='svhn', mnist_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
    
    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
	    tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
	    tf.gfile.MakeDirs(FLAGS.sample_save_path)
    
    if FLAGS.mode == 'pretrain':
	    solver.pretrain()
    elif FLAGS.mode == 'train_sampler':
	    solver.train_sampler()
    elif FLAGS.mode == 'train_dsn':
	    solver.train_dsn()
    elif FLAGS.mode == 'eval_dsn':
	    solver.eval_dsn()
    elif FLAGS.mode == 'test':
	    solver.test()
    elif FLAGS.mode == 'train_convdeconv':
	    solver.train_convdeconv()
    elif FLAGS.mode == 'train_gen_images':
	    solver.train_gen_images()
    
    
    elif FLAGS.mode == 'train_all':
	
	start_img = 0
	end_img = 1600
	
	for alpha in [1.0,0.1,10.0]:
	    for beta in [1.0, 0.1, 10.0]:
		for gamma in [100.,50.,20.,10.,5.]:
	    		
		    model = DSN(mode='train_dsn', learning_rate=0.0001, alpha = alpha, beta = beta, gamma = gamma)
		    solver = Solver(model, svhn_dir='svhn', mnist_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path, start_img = start_img, end_img = end_img)
		    solver.train_dsn()
		    model = DSN(mode='eval_dsn')
		    solver = Solver(model, svhn_dir='svhn', mnist_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
		    solver.eval_dsn(name=str(alpha)+str(beta)+str(gamma))

    else:
	print 'Unrecognized mode.'
	
        
if __name__ == '__main__':
    tf.app.run()



    


