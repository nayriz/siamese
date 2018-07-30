#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:47:53 2018

@author: john
"""

import numpy as np
import tensorflow as tf

X = np.load('data_google/X.npy')
y = np.load('data_google/y.npy')

inds = list(range(len(y)))
np.random.shuffle(inds)

X = X[inds]
y = y[inds]

n_train = 10000
X_train_small = X[:n_train]
y_train_small = y[:n_train]

np.save('data_google/X_train_small.npy',X_train_small)
np.save('data_google/y_train_small.npy',y_train_small)

n_test = 1000
X_test_small = X[:n_test]
y_test_small = y[:n_test]

np.save('data_google/X_test_small.npy',X_test_small)
np.save('data_google/y_test_small.npy',y_test_small)