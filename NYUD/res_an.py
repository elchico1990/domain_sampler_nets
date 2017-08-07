import numpy as np
import cPickle

trg_accs =  dict()

for i in range(1,7):
	with open('trg_acc_'+str(i)+'.pkl') as f:
		trg_accs[str(i)], trg_labels = cPickle.load(f)
		
#~ trg_labels = trg_labels[None,:]		 
trg_accs = np.array(trg_accs.values())

for i in range(6):
	qwe = np.where((trg_labels==trg_accs[i,:]))[0]
	print len(qwe)/float(len(trg_labels))

print '....................................'
qwe = np.where((trg_labels==np.median(trg_accs[2:],0)))[0]
print len(qwe)/float(len(trg_labels))


print 'break'		 

