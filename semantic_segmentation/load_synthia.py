import numpy as np 
import numpy.random as npr
import pandas as pd
from PIL import Image
from scipy import misc

import imageio

import os
import glob

def load_synthia(seq_name, no_elements=1000):
    
    print 'Loading SYNTHIA dataset,',str(no_elements),'samples.'

    data_dir = './data'
    seq_num = '01'  
    mode = 'DAWN'

    #~ img_dir = os.path.join(data_dir,'SYNTHIA-SEQS-'+seq_num+'-'+mode,'RGB/Stereo_Left/Omni_F')

    #~ img_dir = './data/SYNTHIA/Omni_F_RGB'
    #~ gt_labels_dir = './data/SYNTHIA/Omni_F_GT_LABELS'
    
    img_dir = '/cvgl/group/Synthia/'+seq_name+'/RGB/Stereo_Left/Omni_F'
    gt_labels_dir = '/cvgl/group/Synthia/'+seq_name+'/GT/LABELS/Stereo_Left/Omni_F' 

    img_files = sorted(glob.glob(img_dir+'/*'))[:no_elements]
    gt_labels_files = sorted(glob.glob(gt_labels_dir+'/*'))[:no_elements]

    scale = 1

    images = np.zeros((len(img_files), 224 * scale, 224 * scale, 3))
    gt_labels = np.zeros((len(gt_labels_files), 224 * scale, 224 * scale))

    for n, img, gt_lab in zip(range(len(img_files)), img_files, gt_labels_files):
	
	#~ print n
	
	img = misc.imread(img)
	img = misc.imresize(img,(224 * scale, 224 * scale,3))
	
	gt_lab = np.asarray(imageio.imread(gt_lab, format='PNG-FI'))[:,:,0]  # uint16
	gt_lab = misc.imresize(gt_lab,(224 * scale, 224 * scale),interp='nearest') / 17
	
	
	images[n] = img
	gt_labels[n] = gt_lab
    
    gt_labels[gt_labels==15.] = 13.
    
    npr.seed(231)
    
    rnd_indices = np.arange(0,len(images))
    npr.shuffle(rnd_indices)
    
    imags = images[rnd_indices]
    gt_labels = gt_labels[rnd_indices]
    
    return images, np.expand_dims(gt_labels,3).astype(int)

if __name__=='__main__':
    
    images, gt_labels = load_synthia(no_elements=100)
    print 'break'







