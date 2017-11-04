import tensorflow as tf
from model import DSN
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_string('splits', 'amazon2webcam', "src2trg")
FLAGS = flags.FLAGS

def main(_):
    
    model = DSN(mode=FLAGS.mode, learning_rate=0.00005)
    src_split, trg_split = FLAGS.splits.split('2')[0], FLAGS.splits.split('2')[1]
    solver = Solver(model, batch_size=128, src_dir=src_split, trg_dir=trg_split)
    

    
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
    elif FLAGS.mode == 'test_ensemble':
	    solver.test_ensemble()


    else:
	print 'Unrecognized mode.'
	
        
if __name__ == '__main__':
    tf.app.run()











    


