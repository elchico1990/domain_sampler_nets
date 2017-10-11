import numpy as np


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights

#...............................................................................................
#
#
# %matplotlib inline
#
# from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt

import urllib2

slim = tf.contrib.slim

import vgg
import vgg_preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

image_filename = 'cat.jpg'
annotation_filename = 'cat_annotation.png'

no_classes = 3

#plaeholders
image_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')
annotation_tensor = tf.placeholder(tf.float32, [None, 224, 224, 1], 'annotations')
is_training_placeholder = tf.placeholder(tf.bool)




#~ # Get ones for each class instead of a number -- we need that
#~ # for cross-entropy loss later on. Sometimes the groundtruth
#~ # masks have values other than 1 and 0.
#~ class_labels_tensor = tf.equal(annotation_tensor, 1)
#~ background_labels_tensor = tf.not_equal(annotation_tensor, 1)
#~ # Convert the boolean values into floats -- so that
#~ # computations in cross-entropy loss is correct
#~ bit_mask_class = tf.to_float(class_labels_tensor)
#~ bit_mask_background = tf.to_float(background_labels_tensor)
#~ combined_mask = tf.concat(axis=3, values=[bit_mask_class,
                                                #~ bit_mask_background])
#~ # Lets reshape our input so that it becomes suitable for
#~ # tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
#~ flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))

labels_tensors = [tf.to_float(tf.equal(annotation_tensor, i)) for i in range(no_classes)]
combined_mask = tf.concat(axis=3, values = labels_tensors)
flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, no_classes))



#...........................................................................................




fig_size = [15, 4]
plt.rcParams["figure.figsize"] = fig_size



# Load the mean pixel values and the function
# that performs the subtraction from each pixel
from vgg_preprocessing import (_mean_image_subtraction,
                                             _R_MEAN, _G_MEAN, _B_MEAN)

upsample_factor = 32

log_folder = './logs'

vgg_checkpoint_path = './vgg_16.ckpt'

# Convert image to float32 before subtracting the
# mean pixel value
image_float = tf.to_float(image_tensor, name='ToFloat')

# Subtract the mean pixel value from each pixel
#~ mean_centered_image = _mean_image_subtraction(image_float,
                                              #~ [_R_MEAN, _G_MEAN, _B_MEAN])


#~ image_float = tf.expand_dims(image_float, 0)

processed_images = tf.subtract(image_float, tf.constant([_R_MEAN, _G_MEAN, _B_MEAN]))

upsample_filter_np = bilinear_upsample_weights(upsample_factor,
                                               no_classes)

upsample_filter_tensor = tf.constant(upsample_filter_np)

# Define the model that we want to use -- specify to use only two classes at the last layer
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, end_points = vgg.vgg_16(processed_images,
                                    num_classes=no_classes,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='VALID')
				    
    features_fc7 = vgg.vgg_16(processed_images,
                                    num_classes=no_classes,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,
                                    fc_conv_padding='VALID',
				    reuse=True,
				    return_fc7=True)

# First upsampling to match shapes
logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor,
                                          output_shape=[tf.shape(logits)[0],7,7,no_classes],
                                          strides=[1, 7, 7, 1])

downsampled_logits_shape = tf.shape(logits)


# Calculate the ouput size of the upsampled tensor
upsampled_logits_shape = tf.stack([
    downsampled_logits_shape[0],
    downsampled_logits_shape[1] * upsample_factor,
    downsampled_logits_shape[2] * upsample_factor,
    downsampled_logits_shape[3]
])

# Perform the upsampling
upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor,
                                          output_shape=upsampled_logits_shape,
                                          strides=[1, upsample_factor, upsample_factor, 1])

# Flatten the predictions, so that we can compute cross-entropy for
# each pixel and get a sum of cross-entropies.
flat_logits = tf.reshape(tensor=upsampled_logits, shape=(-1, no_classes))

cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                          labels=flat_labels)

cross_entropy_sum = tf.reduce_sum(cross_entropies)

# Tensor to get the final prediction for each pixel -- pay
# attention that we don't need softmax in this case because
# we only need the final decision. If we also need the respective
# probabilities we will have to apply softmax.
pred = tf.argmax(upsampled_logits, dimension=3)

probabilities = tf.nn.softmax(upsampled_logits)

# Here we define an optimizer and put all the variables
# that will be created under a namespace of 'adam_vars'.
# This is done so that we can easily access them later.
# Those variables are used by adam optimizer and are not
# related to variables of the vgg model.

# We also retrieve gradient Tensors for each of our variables
# This way we can later visualize them in tensorboard.
# optimizer.compute_gradients and optimizer.apply_gradients
# is equivalent to running:
# train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy_sum)
with tf.variable_scope("adam_vars"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    gradients = optimizer.compute_gradients(loss=cross_entropy_sum)

    for grad_var_pair in gradients:
        current_variable = grad_var_pair[1]
        current_gradient = grad_var_pair[0]

        # Relace some characters from the original variable name
        # tensorboard doesn't accept ':' symbol
        gradient_name_to_save = current_variable.name.replace(":", "_")

        # Let's get histogram of gradients for each layer and
        # visualize them later in tensorboard
        tf.summary.histogram(gradient_name_to_save, current_gradient)

    train_step = optimizer.apply_gradients(grads_and_vars=gradients)

# Now we define a function that will load the weights from VGG checkpoint
# into our variables when we call it. We exclude the weights from the last layer
# which is responsible for class predictions. We do this because
# we will have different number of classes to predict and we can't
# use the old ones as an initialization.
vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'adam_vars'])

# Here we get variables that belong to the last layer of network.
# As we saw, the number of classes that VGG was originally trained on
# is different from ours -- in our case it is only 2 classes.
vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])

adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

# Add summary op for the loss -- to be able to see it in
# tensorboard.
tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

# Put all summary ops into one op. Produces string when
# you run it.
merged_summary_op = tf.summary.merge_all()

# Create the summary writer -- to write all the logs
# into a specified file. This file can be later read
# by tensorboard.
summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Create an OP that performs the initialization of
# values of variables to the values from VGG.
read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
    vgg_checkpoint_path,
    vgg_except_fc8_weights)

# Initializer for new fc8 weights -- for two classes.
vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

# Initializer for adam variables
optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)












with tf.Session() as sess:
    # Run the initializers.
    read_vgg_weights_except_fc8_func(sess)
    sess.run(vgg_fc8_weights_initializer)
    sess.run(optimization_variables_initializer)

    images = np.zeros((10, 224,224,3))
    annotations = np.zeros((1000, 224,224,1))

    feed_dict = {image_tensor: images,
		    annotation_tensor: annotations,
		    is_training_placeholder: True}


    logits, upsampled_logits, flat_logits, features_fc7, processed_images, train_images, train_annotations = sess.run([logits, upsampled_logits, flat_logits, features_fc7, processed_images, image_tensor, annotation_tensor],
                                             feed_dict=feed_dict)
					     
    print upsampled_logits.shape, upsampled_logits.max(), upsampled_logits.min(), upsampled_logits.mean() 
    print flat_logits.shape, flat_logits.max(), flat_logits.min(), flat_logits.mean() 
					     
    #~ f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #~ ax1.imshow(train_image)
    #~ ax1.set_title('Input image')
    #~ probability_graph = ax2.imshow(np.dstack((train_annotation,) * 3) * 100)
    #~ ax2.set_title('Input Ground-Truth Annotation')
    #~ plt.show()

    EPOCHS = 10
    BATCH_SIZE = 2

    for e in range(EPOCHS):
	
	for start, end in zip(range(0,len(images),BATCH_SIZE), range(BATCH_SIZE,len(images),BATCH_SIZE)):
	    
		    
	    feed_dict = {image_tensor: images[start:end], annotation_tensor: annotations[start:end], is_training_placeholder: True}

	    
	    loss, summary_string = sess.run([cross_entropy_sum, merged_summary_op], feed_dict=feed_dict)

	    sess.run(train_step, feed_dict=feed_dict)

	    pred_np, probabilities_np = sess.run([pred, probabilities],	feed_dict=feed_dict)

	    summary_string_writer.add_summary(summary_string, e)

	    #~ cmap = plt.get_cmap('bwr')

	    #~ f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	    #~ ax1.imshow(np.uint8(pred_np.squeeze() != 1), vmax=1.5, vmin=-0.4, cmap=cmap)
	    #~ ax1.set_title('Argmax. Iteration # ' + str(i))
	    #~ probability_graph = ax2.imshow(probabilities_np.squeeze()[:, :, 0])
	    #~ ax2.set_title('Probability of the Class. Iteration # ' + str(i))

	    #~ plt.colorbar(probability_graph)
	    #~ plt.show()

	    print("Current Loss: " + str(loss))

    feed_dict[is_training_placeholder] = False

    final_predictions, final_probabilities, final_loss = sess.run([pred,
                                                                   probabilities,
                                                                   cross_entropy_sum],
                                                                  feed_dict=feed_dict)

    #~ f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    #~ ax1.imshow(np.uint8(final_predictions.squeeze() != 1),
               #~ vmax=1.5,
               #~ vmin=-0.4,
               #~ cmap=cmap)

    #~ ax1.set_title('Final Argmax')

    #~ probability_graph = ax2.imshow(final_probabilities.squeeze()[:, :, 0])
    #~ ax2.set_title('Final Probability of the Class')
    #~ plt.colorbar(probability_graph)

    #~ plt.show()

    print("Final Loss: " + str(final_loss))

summary_string_writer.close()



