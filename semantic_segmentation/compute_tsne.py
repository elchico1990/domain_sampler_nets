import cPickle
from sklearn.manifold import TSNE

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

file_dir = './experiments/SYNTHIA-SEQS-01-DAWN/features_dsn.pkl'

with open(file_dir,'r') as f:
	source_features, target_features, generated_features = cPickle.load(f)

generated_features = generated_features[0]

model = TSNE(n_components=2, random_state=0)

f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

ax1.set_facecolor('white')
ax2.set_facecolor('white')
ax3.set_facecolor('white')
	    
print 'Compute t-SNE 1.'
TSNE_hA = model.fit_transform(np.vstack((source_features, generated_features)))
ax1.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((len(generated_features),)), 2 * np.ones((len(source_features),)))), s=3, cmap = mpl.cm.jet, alpha=0.5)

print 'Compute t-SNE 2.'
TSNE_hA = model.fit_transform(np.vstack((source_features, target_features['SYNTHIA-SEQS-01-FALL'])))
ax2.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((len(generated_features),)), 2 * np.ones((len(source_features),)))), s=3, cmap = mpl.cm.jet, alpha=0.5)


print 'Compute t-SNE 3.'
TSNE_hA = model.fit_transform(np.vstack((source_features, target_features['SYNTHIA-SEQS-01-DAWN'])))
ax3.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((len(generated_features),)), 2 * np.ones((len(source_features),)))), s=3, cmap = mpl.cm.jet, alpha=0.5)

plt.title('NIGHT vs GENERATED - NIGHT vs FALL - NIGHT vs DAWN')
plt.show()
