import matplotlib.pyplot as plt
import cPickle
import numpy as np


file_FA = './usps_mnist_sdjfn.pkl' 
file_di_ADDA = './usps_mnist_Training with domain-invariant ADDA algorithm..pkl' 
file_ADDA = './usps_mnist_adda.pkl'

with open(file_FA,'r') as f:
	data_FA = cPickle.load(f)
with open(file_di_ADDA,'r') as f:
	data_di_ADDA = cPickle.load(f)
with open(file_ADDA,'r') as f:
	data_ADDA = cPickle.load(f)

data_FA = np.array(data_FA)
data_di_ADDA = np.array(data_di_ADDA)
data_ADDA = np.array(data_ADDA)

print 'ADDA:', str(data_ADDA[-1])
print 'Domain-invariant ADDA:', str(data_di_ADDA[-1])
print 'Feature Augmentation:', str(data_FA[-1])

plt.plot(data_FA)
plt.plot(data_di_ADDA)
plt.plot(data_ADDA)
plt.show()

