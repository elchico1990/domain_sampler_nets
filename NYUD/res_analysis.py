import cPickle
import matplotlib.pyplot as plt

data_dir = './'
with open(data_dir + 'train_dsn_test_accuracies.pkl','rb') as f:
    qwe = cPickle.load(f)
plt.plot(qwe)

data_dir = './'
with open(data_dir + 'train_adda_shared_test_accuracies.pkl','rb') as f:
    qwe = cPickle.load(f)
plt.plot(qwe)

plt.ylabel('Accuracy %')
plt.xlabel('Iterations (x20)')

plt.show()

