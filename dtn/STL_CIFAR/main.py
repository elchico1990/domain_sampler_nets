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
	
	start_img = 1600
	end_img = 3200
	
	for start,end,name in zip([8000,9600,11200,12800,14400,16000,17600,19200,20800,22400,24000,25600],
							  [9600,11200,12800,14400,16000,17600,19200,20800,22400,24000,25600,27200],
							  ['Exp6','Exp7','Exp8','Exp9','Exp10','Exp11','Exp12','Exp13','Exp14','Exp15','Exp16','Exp17']):
	
	    model = DSN(mode='train_dsn', learning_rate=0.0001)
	    solver = Solver(model, svhn_dir='svhn', mnist_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path, start_img = start_img, end_img = end_img)
	    solver.train_dsn()
	    
	    model = DSN(mode='eval_dsn')
	    solver = Solver(model, svhn_dir='svhn', mnist_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
	    solver.eval_dsn(name=name)

	    tf.reset_default_graph()

    else:
	print 'Unrecognized mode.'
	
        
if __name__ == '__main__':
    tf.app.run()



    


