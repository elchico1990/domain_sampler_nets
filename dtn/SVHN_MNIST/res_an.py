import matplotlib.pyplot as plt
import cPickle
import numpy as np


file_FA = './svhn_mnist_fa.pkl' 
file_di_ADDA = './svhn_mnist_adda_di.pkl' 
file_ADDA = './svhn_mnist_adda.pkl'

with open(file_FA,'r') as f:
	data_FA = cPickle.load(f)
with open(file_di_ADDA,'r') as f:
	data_di_ADDA = cPickle.load(f)
with open(file_ADDA,'r') as f:
	data_ADDA = cPickle.load(f)

data_FA = np.array(data_FA)
data_di_ADDA = np.array(data_di_ADDA)
data_ADDA = np.array(data_ADDA)

print 'ADDA:', str(data_ADDA[-10:].mean())
print 'Domain-invariant ADDA:', str(data_di_ADDA[-10:].mean())
print 'Feature Augmentation:', str(data_FA[-10:].mean())

plt.plot(data_FA, 'g')
plt.plot(data_di_ADDA, 'r')
plt.plot(data_ADDA, 'b')
plt.show()

