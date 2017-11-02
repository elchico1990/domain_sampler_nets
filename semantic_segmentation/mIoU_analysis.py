import numpy as np
import cPickle
import matplotlib.pyplot as plt

from load_synthia import load_synthia

with open('./mIoU_analysis.pkl','r') as f:
    source_images, source_annotations, source_preds = cPickle.load(f)

source_annotations = np.squeeze(source_annotations)

source_images, source_annotations = load_synthia('SYNTHIA-SEQS-01-NIGHT', no_elements=10)
	

pred = source_preds[1]
ann = source_annotations[1]
img - source_images[1]

plt.figure(0)
plt.imshow(pred)
plt.figure(1)
plt.imshow(ann)
plt.figure(2)
plt.imshow(img)

plt.plot()

print 'break'


