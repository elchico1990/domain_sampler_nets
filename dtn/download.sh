mkdir -p mnist
mkdir -p svhn
mkdir -p usps
mkdir -p office

wget -O data/svhn/train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -O data/svhn/test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget -O data/usps/USPS.mat http://www.cad.zju.edu.cn/home/dengcai/Data/USPS/USPS.mat
