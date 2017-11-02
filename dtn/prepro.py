import numpy as np
import pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
import glob

def resize_images(image_arrays, size=[32,32]):
    # convert float type to integer 
    image_arrays = (image_arrays * 255).astype('uint8')
    
    resized_image_arrays = np.zeros([image_arrays.shape[0]]+size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.ANTIALIAS)
        
        resized_image_arrays[i] = np.asarray(resized_image)
    
    return np.expand_dims(resized_image_arrays, 3)  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

def main():
    mnist = input_data.read_data_sets(train_dir='mnist')

    train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),
             'y': mnist.train.labels}
    
    test = {'X': resize_images(mnist.test.images.reshape(-1, 28, 28)),
            'y': mnist.test.labels}
        
    save_pickle(train, 'data/mnist/train.pkl')
    save_pickle(test, 'data/mnist/test.pkl')

def usps():
	

	uspsData = scipy.io.loadmat('./data/usps/USPS.mat')
	
	images = (uspsData['fea'] + 1)/2.
	
	images = images.reshape(-1,16,16)

	images = resize_images(images)

	labels = uspsData['gnd']
	labels[np.where(labels==10)] = 0

	train = {'X': images, 'y': labels}

	save_pickle(train, 'data/usps/train.pkl')

def office():
	
	import cv2

	dataDir = './office/'
	subFolders = ['amazon','webcam','dslr']
	#~ subFolders = ['webcam','dslr']
	mean = np.array([104., 117., 124.])
	 
	#for each domain
	for sf in subFolders:
	    
	    print sf
	    objects = glob.glob(dataDir + sf + '/images/*')
	    counter = 0
	    images_list = []
	    labels_list = []
	    #for each class
	    for obj in objects:
		#image list
		images_list += glob.glob(obj + '/*')
		for elem in glob.glob(obj + '/*'):
		    labels_list += [counter]
		counter += 1
	    assert len(images_list) == len(labels_list)
	    
	    images = np.zeros((len(images_list),227,227,3))
	    labels = np.asarray(labels_list,dtype=np.int32)
	    #~ print labels
	    #~ print images_list
	    
	    for ii, im_path in enumerate(images_list):
		img = cv2.imread(im_path)
		img = cv2.resize(img, (227, 227))
		img = img.astype(np.float32)
		#subtract mean
		img -= mean
		images[ii,:,:,:] = img
		
	    if sf == 'amazon':
		train1 = {'X': images[:len(images_list)/2], 'y': labels[:len(images_list)/2]}
		train2 = {'X': images[len(images_list)/2:], 'y': labels[len(images_list)/2:]}
		save_pickle(train1, 'office/'+sf+'_1.pkl')
		save_pickle(train2, 'office/'+sf+'_2.pkl')
	    else:
		train = {'X': images, 'y': labels}
		save_pickle(train, 'office/'+sf+'.pkl')


if __name__ == "__main__":
    main()
    #~ usps()
    #~ office()
    
