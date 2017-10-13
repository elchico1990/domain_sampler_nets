import numpy as np 
import pandas as pd
from PIL import Image
from scipy import misc

import os
import glob

def load_synthia(no_elements=1000):
    
    print 'Loading SYNTHIA dataset,',str(no_elements),'samples.'

    data_dir = './data'
    seq_num = '01'  
    mode = 'DAWN'

    #~ img_dir = os.path.join(data_dir,'SYNTHIA-SEQS-'+seq_num+'-'+mode,'RGB/Stereo_Left/Omni_F')

    img_dir = './data/SYNTHIA_RAND_CVPR16/RGB'
    gt_dir = './data/SYNTHIA_RAND_CVPR16/GTTXT'

    img_files = sorted(glob.glob(img_dir+'/*'))[:no_elements]
    gt_files = sorted(glob.glob(gt_dir+'/*'))[:no_elements]


    images = np.zeros((len(img_files),704,960,3))
    labels = np.zeros((len(gt_files), 704,960, 1))

    for n, img, lab in zip(range(len(img_files)), img_files, gt_files):
	
	#~ if n%50==0:
	    #~ print n
	
	#~ print lab
	
	img = misc.imread(img)
	img = np.resize(img,(704,960,3))
	
	lab = np.array(pd.read_csv(lab, ' ', header=None))
	lab[lab==-1] = 12
	lab = np.resize(lab,(704,960,1))

	
	images[n] = img
	labels[n] = lab
	
    return images, labels


if __name__=='__main__':
    
    images, labels = load_synthia(no_elements=50)
    
    print 'break'







