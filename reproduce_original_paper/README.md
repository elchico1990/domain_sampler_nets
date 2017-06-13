# Domain Transfer Network (DTN) 
TensorFlow implementation of [Unsupervised Cross-Domain Image Generation.](https://arxiv.org/abs/1611.02200)
![alt text](../dtn/jpg/dtn.jpg)


#### Orignal code [here](https://github.com/yunjey/dtn-tensorflow).

## Usage

#### Download the dataset
```bash
$ chmod +x download.sh
$ ./download.sh
```

#### Resize MNIST dataset to 32x32 
```bash
$ python prepro.py
```

#### Pretrain the model f
```bash
$ python main.py --mode='pretrain'
```

#### Train the model G and D
```bash
$ python main.py --mode='train'
```

#### Transfer SVHN to MNIST
```bash
$ python main.py --mode='eval'
```
