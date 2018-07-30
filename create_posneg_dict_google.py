#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:31:49 2018

@author: john
"""

import numpy as np
import json
X_train = np.load('data_google/X_train_small.npy')
y_train = np.load('data_google/y_train_small.npy')

#X_train = np.load('data_google/X.npy')
#y_train = np.load('data_google/y.npy')

n_train = X_train.shape[0]

classes = list(range(np.amax(y_train)))


#np.random.shuffle(classes)
train_classes = classes[:np.amax(y_train)]
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
    
with open('data_google/positive_train.json', 'w') as f:
    json.dump(inds_positive, f,default=default)    
f.close()

# initialize the dictionary of indices of all negative 
# samples corresonding to a class
inds_all_negative = {}
#ind_train = np.array(range(n_train))

for i in train_classes:       
    
    inds_all_negative[int(i)] = list(np.where(y_train != i)[0])

with open('data_google/negative_train.json', 'w') as f:
    json.dump(inds_all_negative, f,default=default)    
f.close()

#######################################################################
#X_test = np.load('data_google/X.npy')
#y_test = np.load('data_google/y.npy')

X_test = np.load('data_google/X_test_small.npy')
y_test = np.load('data_google/y_test_small.npy')


n_test = X_test.shape[0]
classes = list(range(np.amax(y_train)))
#np.random.shuffle(classes)
test_classes = list(range(np.amax(y_train)))
inds_positive = {}
INDS_anchor = []
inds_positive = {}
INDS_positive = []

def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

json.dumps({'value': np.int64(42)}, default=default)

# initialize the total number of testing samples
N_test = 0

for i in test_classes:
    
    # find the indices each class in the testing set
    ind_i = list(np.where(y_test == i)[0])

    # save the indices for each class
    inds_positive[int(i)] = ind_i
    
with open('data_google/positive_test.json', 'w') as f:
    json.dump(inds_positive, f,default=default)    
f.close()

# initialize the dictionary of indices of all negative 
# samples corresonding to a class
inds_all_negative = {}
#ind_test = np.array(range(n_test))

for i in test_classes:       
    
    inds_all_negative[int(i)] = list(np.where(y_test != i)[0])

with open('data_google/negative_test.json', 'w') as f:
    json.dump(inds_all_negative, f,default=default)    
f.close()  