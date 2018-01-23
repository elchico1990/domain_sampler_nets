import cPickle
import matplotlib.pyplot as plt

data_dir = './'
try:
    f = open(data_dir + 'train_adda_test_accuracies.pkl','rb')
    qwe = cPickle.load(f)
    plt.plot(qwe)
except:
    print 'train_adda_test_accuracies.pkl not found '

try:
    f = open(data_dir + 'train_adda_shared_test_accuracies.pkl','rb')
    qwe = cPickle.load(f)
    plt.plot(qwe)
except:
    print 'train_adda_shared_test_accuracies.pkl not found '

try:
    f = open(data_dir + 'train_dsn_test_accuracies.pkl','rb')
    qwe = cPickle.load(f)
    plt.plot(qwe)
except:
    print 'train_dsn_test_accuracies.pkl not found '


plt.ylabel('Accuracy %')
plt.xlabel('Iterations (x50)')

plt.show()

