#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 22:04:54 2018

@author: john
"""

import json
import numpy as np

with open('MNIST/positive_train.json') as f:   
    positive_train = json.load(f)
f.close()

with open('MNIST/negative_train.json') as f:   
    negative_train = json.load(f)
f.close()

X_train = np.load('MNIST/X_train_MNIST.npy')
y_train = np.load('MNIST/y_train_MNIST.npy')

classes = [0, 1]

#y_anchor = []
#y_positive = []
#y_negative = []

y_epoch1 = []
y_epoch2 = []

test = [5, 7]

for i in classes :

    y = positive_train[str(i)]
    y_neg = negative_train[str(i)] 
    np.random.shuffle(y)
    np.random.shuffle(y_neg)    
    n_y = len(y)
    
    y_epoch1 += y[:2*int(n_y/3)]
    yp = y[2*int(n_y/3):3*int(n_y/3)]
    yn = y[:int(n_y/3)]

    y_epoch2 += yp  
    y_epoch2 += yn
    
    
#    y_anchor += ya
#    y_positive += yp
#    y_negative += yn

    

