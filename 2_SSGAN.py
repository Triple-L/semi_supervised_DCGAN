#完成数据之后
#用terminal 或者 在pycharm中设置参数进行代码运行
#参考 READMEN文件
#注意！ YOUR_DATASETS不太好使

'''
可以跑通一个然后修改 比如Mnist 或者Cifar-10

Mnist 28*28 OK! 用新代码 不是本文中的这些 重新开一个工程

重新resize数据 28*28
'''

'''Download datasets with:

$ python download.py --dataset MNIST SVHN CIFAR10
Train models with downloaded datasets:

$ python trainer.py --dataset MNIST
$ python trainer.py --dataset SVHN
$ python trainer.py --dataset CIFAR10
Test models with saved checkpoints:

$ python evaler.py --dataset MNIST --checkpoint ckpt_dir
$ python evaler.py --dataset SVHN --checkpoint ckpt_dir
$ python evaler.py --dataset CIFAR10 --checkpoint ckpt_dir
The ckpt_dir should be like: train_dir/default-MNIST_lr_0.0001_update_G5_D1-20170101-194957/model-1001

Train and test your own datasets:

Create a directory
$ mkdir datasets/YOUR_DATASET
Store your data as an h5py file datasets/YOUR_DATASET/data.hy and each data point contains
'image': has shape [h, w, c], where c is the number of channels (grayscale images: 1, color images: 3)
'label': represented as an one-hot vector
Maintain a list datasets/YOUR_DATASET/id.txt listing ids of all data points
Modify trainer.py including args, data_info, etc.
Finally, train and test models:
$ python trainer.py --dataset YOUR_DATASET
$ python evaler.py --dataset YOUR_DATASET'''