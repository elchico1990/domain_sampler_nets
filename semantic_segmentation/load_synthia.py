import numpy as np 
import pandas as pd
from PIL import Image

import os
import glob

def load_synthia(no_elements=1000):
    
    print 'Loading SYNTHIA dataset,',str(no_elements),'samples.'

    data_dir = './data'
    seq_num = '01'  
    mode = 'DAWN'
    size = (224,224)

    #~ img_dir = os.path.join(data_dir,'SYNTHIA-SEQS-'+seq_num+'-'+mode,'RGB/Stereo_Left/Omni_F')

    img_dir = './data/SYNTHIA_RAND_CVPR16/RGB'
    gt_dir = './data/SYNTHIA_RAND_CVPR16/GTTXT'

    img_files = sorted(glob.glob(img_dir+'/*'))[:no_elements]
    gt_files = sorted(glob.glob(gt_dir+'/*'))[:no_elements]


    images = np.zeros((len(img_files),720,960,3))
    labels = np.zeros((len(gt_files), 720, 960, 1))

    for n, img, lab in zip(range(len(img_files)), img_files, gt_files):
	
	if n%50==0:
	    print n
	
	img = Image.open(img)
	#~ img = img.resize(size=size, resample=Image.ANTIALIAS)
	
	lab = np.array(pd.read_csv(lab, ' '))
	lab[lab==-1] = 12
	#~ lab = np.resize(lab,size)
	lab = np.expand_dims(lab,2)
	
	images[n] = np.asarray(img)
	labels[n] = lab
	
    return images, labels


if __name__=='__main__':
    
    images, labels = load_synthia(no_elements=50)
    
    print 'break'







