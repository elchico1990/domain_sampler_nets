import numpy as np
import pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io
import glob

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)

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

    office()
    
