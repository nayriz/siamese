#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:15:28 2018

@author: john
"""

import imageio
import os
path1 = '/media/john/github/siamese/omniglot/python/images_background/'

listdir1 = os.listdir(path1)

listdir1.sort()


train_list = []
for alphabet in listdir1 :
    
    listdir_tmp = os.listdir(path1 + alphabet)
    
    listdir_tmp.sort()
    
    for letter in listdir_tmp:
        
        path2 = path1 + '/'+ alphabet + '/' + letter + '/'
        filelist = os.listdir(path2)
        filelist.sort()
            
        for sample in filelist :
            

        
            img = imageio.imread(path2 + sample)
            train_list.append(img)

    
#im = imageio.imread('my_image.png')
#X_train = np.load('MNIST/X_train_MNIST.npy')
#y_train = np.load('MNIST/y_train_MNIST.npy')
#X_test = np.load('MNIST/X_test_MNIST.npy')
#y_test = np.load('MNIST/y_test_MNIST.npy')