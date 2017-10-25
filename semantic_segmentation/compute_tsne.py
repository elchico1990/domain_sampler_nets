import cPickle
from sklearn.manifold import TSNE

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

file_dir = './experiments/SYNTHIA-SEQS-01-DAWN/features.pkl'

with open(file_dir,'r') as f:
	source_features, target_features, generated_features = cPickle.load(f)

generated_features = generated_features[0]

model = TSNE(n_components=2, random_state=0)
TSNE_hA = model.fit_transform(np.vstack((target_features,source_features)))

plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((len(generated_features),)), 40 * np.ones((len(source_features),)))), s=6, cmap = mpl.cm.flag, alpha=0.9)
plt.show()
