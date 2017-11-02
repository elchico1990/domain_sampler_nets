import tensorflow as tf
import numpy as np

# AlexNet Avg Baselines
#
# AW  60.6
# DW  95.4
# WD  99.0
# AD  64.2
# DA  45.5
# WA  48.3
# Avg 68.8

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, is_training, keep_prob_input, keep_prob_conv, keep_prob_hidden, num_classes, skip_layer,
                 weights_path='DEFAULT',reuse=False, hidden_repr_size = 128):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB_INPUT = keep_prob_input
        self.KEEP_PROB_CONV = keep_prob_conv
        self.KEEP_PROB_HIDDEN = keep_prob_hidden
        self.SKIP_LAYER = skip_layer
	self.HIDDEN_REPR_SIZE = hidden_repr_size
	self.is_training = is_training

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create(reuse=reuse)


    def create(self,reuse=False):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
	self.X = dropout(self.X,self.KEEP_PROB_INPUT)
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1',reuse=reuse)
	conv1 = dropout(conv1, self.KEEP_PROB_CONV)
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2',reuse=reuse)
	conv2 = dropout(conv2, self.KEEP_PROB_CONV)
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3',reuse=reuse)
	conv3 = dropout(conv3, self.KEEP_PROB_CONV)
	#~ conv3 = tf.contrib.layers.batch_norm(conv3,self.is_training,center=True,scale=True,scope='bna3',reuse=reuse)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4',reuse=reuse)
	conv4 = dropout(conv4, self.KEEP_PROB_CONV)
	#~ conv4 = tf.contrib.layers.batch_norm(conv4,self.is_training,center=True,scale=True,scope='bna4',reuse=reuse)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5',reuse=reuse)
	conv5 = dropout(conv5, self.KEEP_PROB_CONV)
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6',reuse=reuse,is_training=self.is_training)
        dropout6 = dropout(fc6, self.KEEP_PROB_HIDDEN)
	#~ dropout6 = tf.contrib.layers.batch_norm(dropout6,self.is_training,center=True,scale=True,scope='bna6',reuse=reuse)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7',reuse=reuse,is_training=self.is_training)
        dropout7 = dropout(fc7, self.KEEP_PROB_HIDDEN)
	#~ dropout7 = tf.contrib.layers.batch_norm(dropout7,self.is_training,center=True,scale=True,scope='bna7',reuse=reuse)

        
        self.fc_repr = fc(dropout7, 4096, self.HIDDEN_REPR_SIZE, tanh=True, name='fc_repr',reuse=reuse,is_training=self.is_training)
        dropout_repr =  dropout(self.fc_repr, self.KEEP_PROB_HIDDEN)
	#~ dropout_repr = tf.contrib.layers.batch_norm(dropout_repr,self.is_training,center=True,scale=True,scope='bna_repr',reuse=reuse)

        

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(dropout_repr, self.HIDDEN_REPR_SIZE, self.NUM_CLASSES, relu=False, name='fc8',reuse=reuse,is_training=self.is_training)

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
	print 'Loading AlexNet weights.'
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=True)
                            session.run(var.assign(data))
			    #~ print 'loaded bias'

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=True)
                            session.run(var.assign(data))
			    #~ print 'loaded weight'

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1,reuse=False):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name,reuse=reuse) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.nn.bias_add(conv, biases)

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True, tanh=False, reuse=False,is_training=True):
    """Create a fully connected layer."""
    if tanh:
        relu=False
        
    with tf.variable_scope(name,reuse=reuse) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', [num_out], trainable=True, initializer=tf.contrib.layers.xavier_initializer())

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases)
        act = tf.contrib.layers.batch_norm(act,is_training,center=True,scale=True,scope=scope.name+'/bn',reuse=reuse)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    elif tanh:
        tanh = tf.tanh(act)
        return tanh
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
