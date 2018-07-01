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

X_train_small = X[:10000]
y_train_small = y[:1000]

np.save('data_google/X_train_small.npy',X_train_small)
np.save('data_google/y_train_small.npy',y_train_small)

X_test_small = X[:5000]
y_test_small = y[:5000]

np.save('data_google/X_test_small.npy',X_test_small)
np.save('data_google/y_test_small.npy',y_test_small)