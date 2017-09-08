import matplotlib.pyplot as plt

with open('./plots/adda_baseline/dsn_train_converging.txt', 'r') as f:
	ll = f.readlines()

accuracies=[]
D = []
G = []
for i, line in enumerate(ll):
	print i
	if 'trg acc' in line:
		acc = line.split('[')[1]
		accuracies.append(acc[:-2])
	if 'Step:' in line:
		d = line.split('DE: [')[1]
		D.append(d[:6])
		g = line.split('E: [')[1]
		G.append(g[:6])

plt.subplot(3,1,1)
plt.plot(D)
plt.ylabel('Discriminator Loss')
plt.xlabel('Iterations (x10)')

plt.subplot(3,1,2)
plt.plot(G)
plt.ylabel('Generator Loss')
plt.xlabel('Iterations (x10)')

plt.subplot(3,1,3)
plt.plot(accuracies)
plt.ylabel('Test Accuracy')
plt.xlabel('Iterations (x20)')

plt.show()
