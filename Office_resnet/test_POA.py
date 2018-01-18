import numpy as np

feat = np.fromfile('features.npy')

for i, f in enumerate(feat):
	feat[i] = 0 if f<0 else 1
	

feat = feat.reshape(2186,128)
print feat

uniques = feat[0]
uniques = np.expand_dims(uniques, axis=0)

for i, sample in enumerate(feat):
	isthere = False
	for u in uniques:
		isthere = isthere or np.array_equal(u, sample)
	if not isthere:
		uniques=np.concatenate((uniques, np.expand_dims(sample, axis=0)))
