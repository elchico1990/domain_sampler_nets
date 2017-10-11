import numpy as np 
from PIL import Image

import os
import glob

data_dir = './data'
seq_num = '01'  
mode = 'DAWN'
size = (224,224)

img_dir = os.path.join(data_dir,'SYNTHIA-SEQS-'+seq_num+'-'+mode,'RGB/Stereo_Left/Omni_F')

img_files = sorted(glob.glob(img_dir+'/*'))

images = np.zeros((len(img_files),224,224,3))

for n,img in enumerate(img_files):
    print n
    img = Image.open(img)
    img = img.resize(size=size, resample=Image.ANTIALIAS)
    images[n] = np.asarray(img)








