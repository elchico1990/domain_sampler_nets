import cPickle
import matplotlib.pyplot as plt

data_dir = './'
with open(data_dir + 'test_acc.pkl','rb') as f:
    qwe = cPickle.load(f)
    plt.plot(qwe)

data_dir = './model/SYN_SVHN_92.37perc_commit_4a89e99ca538c90c2548db3355ff37534545898f/'
with open(data_dir + 'test_acc.pkl','rb') as f:
    qwe = cPickle.load(f)
    plt.plot(qwe[2:])


plt.show()

