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
    solver = Solver(model, src_dir='amazon', trg_dir='dslr', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
    
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
    
    
    elif FLAGS.mode == 'train_all':		
	#~ model = DSN(mode='pretrain', learning_rate=0.0003)
	#~ solver = Solver(model, src_dir='svhn', trg_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
	#~ solver.pretrain()
	#~ model = DSN(mode='train_sampler', learning_rate=0.0003)
	#~ solver = Solver(model, src_dir='svhn', trg_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
	#~ solver.train_sampler()
	model = DSN(mode='train_dsn', learning_rate=0.0001)
	solver = Solver(model, src_dir='svhn', trg_dir='mnist', model_save_path=FLAGS.model_save_path, sample_save_path=FLAGS.sample_save_path)
	solver.train_dsn()

    else:
	    print 'Unrecognized mode.'
	
        
if __name__ == '__main__':
    tf.app.run()



    


