#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:31:49 2018

@author: john
"""

import numpy as np
import json
X_train = np.load('MNIST/X_train_MNIST.npy')
y_train = np.load('MNIST/y_train_MNIST.npy')


n_train = X_train.shape[0]
classes = list(range(10))
#np.random.shuffle(classes)
train_classes = classes[:10]
inds_positive = {}
INDS_anchor = []
inds_positive = {}
INDS_positive = []

def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

json.dumps({'value': np.int64(42)}, default=default)

###############################################################################
for i in train_classes:
    
    # find the indices each class in the training set
    ind_i = list(np.where(y_train == i)[0])

    # save the indices for each class
    inds_positive[int(i)] = ind_i
    
with open('MNIST/positive_train.json', 'w') as f:
    json.dump(inds_positive, f,default=default)    
f.close()

# initialize the dictionary of indices of all negative 
# samples corresonding to a class
inds_all_negative = {}
#ind_train = np.array(range(n_train))

for i in train_classes:       
    
    inds_all_negative[int(i)] = list(np.where(y_train != i)[0])

with open('MNIST/negative_train.json', 'w') as f:
    json.dump(inds_all_negative, f,default=default)    
f.close()
