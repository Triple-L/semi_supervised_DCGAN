#coding=utf-8
#!/usr/bin/python


from __future__ import print_function
import os
import tarfile
import subprocess
import argparse
import h5py
import numpy as np

import os
import sys
from PIL import Image
import os, cv2
from pickled import *
from load_data import *




image_dir="./Lung_data_train/"
resized_dir= "./Resized_train/"  #
lst_file_outdir = "./images.lst"
dim = 28
save_path = './save_train_path/'


image_dir_test ="./test_data/"
resized_dir_test= "./resized_test/"  #
lst_file_outdir_test = "./img_test.lst"
dim = 28
save_path = './save_test_path/'
bin_num=1







def resize_pic(image_dir,resized_dir):

    out_size = (dim, dim)

    filenames = [x for x in os.listdir(image_dir) if not x.startswith('.')]
    for image_name in filenames:
        filename = os.path.join(image_dir, image_name)
        img = Image.open(filename)
        # img.thumbnail(out_size, Image.ANTIALIAS)
        img = img.resize(out_size, Image.BILINEAR)
        out_file = os.path.join(resized_dir, image_name)
        img.save(out_file, "png")

#download.py 改造版




parser = argparse.ArgumentParser(description='Download dataset for SSGAN.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+', choices=['CIFAR10', 'SVHN', 'CIFAR10'])


def prepare_h5py(train_image, train_label, test_image, test_label, data_dir, shape=None):

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
    label = np.concatenate((train_label, test_label), axis=0).astype(np.uint8)

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
    data_id = open(os.path.join(data_dir,'id.txt'), 'w')
    for i in range(image.shape[0]):

        if i%(image.shape[0]/10)==0: #100->10
            bar.update(i/(image.shape[0]/10))

        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        label_vec = np.zeros(2)
        label_vec[label[i]%2] = 1 # 2 CLASS
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()
    return

def check_file(data_dir):
    if os.path.exists(data_dir):
        if os.path.isfile(os.path.join('data.hy')) and \
            os.path.isfile(os.path.join('id.txt')):
            return True
    else:
        os.mkdir(data_dir)
    return False

def data_process_h5py(download_path):
    data_dir=download_path

    def unpickle(file):
        import cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict
    #print(dict.shape)
    num_cifar_train = 165
    num_cifar_test = 165

    target_path = os.path.join(data_dir)
    train_image = []
    train_label = []
    for i in range(bin_num):
        fd = os.path.join(target_path, 'data_batch_'+str(i))
        dict = unpickle(fd)
        train_image.append(dict['data'])
        train_label.append(dict['labels'])

    train_image = np.reshape(np.stack(train_image, axis=0), [num_cifar_train, 32*32*3])
    train_label = np.reshape(np.array(np.stack(train_label, axis=0)), [num_cifar_train])

    fd = os.path.join(target_path, 'test_batch')
    dict = unpickle(fd)
    test_image = np.reshape(dict['data'], [num_cifar_test, 32*32*3])
    test_label = np.reshape(dict['labels'], [num_cifar_test])

    prepare_h5py(train_image, train_label, test_image, test_label, data_dir, [32, 32, 3])




if __name__ == '__main__':

#代码未解耦合&判存，每次仅active其中一步 其余注释掉
#1
    # resize_pic(image_dir,resized_dir) #train
    # resize_pic(image_dir_test,resized_dir_test)#test
#2. for Lung experiment I re-write the code in Lab's computer and get the .lst files
    ####Make_pic_list(resized_dir, lst_file_outdir) # train
# #3.
    data, label, lst = read_data(lst_file_outdir, resized_dir, shape=dim)
    pickled(save_path, data, label, lst_file_outdir, bin_num)
# # # 4.
    args = parser.parse_args()
    if not os.path.exists(save_path): os.mkdir(save_path)
    data_process_h5py(save_path)

#本文件用于封装数据 最后获得 data.hy 和 id.txt 作为SSGAN网络的输入文件