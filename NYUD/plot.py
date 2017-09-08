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

plt.plot()
plt.show()
