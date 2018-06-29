#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:51:50 2018

@author: john
"""

import numpy as np
import tensorflow as tf
import math
from tensorflow.python import debug as tf_debug

X_train = np.load('MNIST/X_train_MNIST.npy')
y_train = np.load('MNIST/y_train_MNIST.npy')
X_test = np.load('MNIST/X_test_MNIST.npy')
y_test = np.load('MNIST/y_test_MNIST.npy')

X_train = np.reshape(X_train,(X_train.shape[0],28*28))
X_test = np.reshape(X_test,(X_test.shape[0],28*28))
batch_size = 64
n_train = X_train.shape[0]
n_test = X_test.shape[0]

classes = list(range(10))
np.random.shuffle(classes)

train_classes = classes[:8]
test_classes = classes[8:10]

list_train = []
for i in train_classes:    
    list_train += np.ndarray.tolist(np.where(y_train == i)[0])
    
list_test = []
for i in test_classes:    
    list_test += np.ndarray.tolist(np.where(y_test == i)[0])

X_train = X_train[list_train]
y_train = y_train[list_train]

X_test = X_test[list_test]
y_test = y_test[list_test]

n_train = X_train.shape[0]
n_test = X_test.shape[0]


inds_train = range(n_train)
inds_test = range(n_test)

batch_size_train = 16
batch_size_test = 16

n_batch_train = int(n_train/batch_size_train)
n_batch_test = int(n_test/batch_size_test)

inds_train = list(range(n_train))
inds_test = list(range(n_test))

inds_test1 = inds_test.copy()
inds_test2 = inds_test.copy()

np.random.shuffle(inds_test1)
np.random.shuffle(inds_test2)
n_epoch = 1000
###############################################################################
tf.reset_default_graph()
sess = tf.Session()
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "talisol:7000")
#X1 = tf.placeholder(tf.float32,[None,28,28,1])
#X2 = tf.placeholder(tf.float32,[None,28,28,1])
X1 = tf.placeholder(tf.float32,[None,28*28],name='X1')
X2 = tf.placeholder(tf.float32,[None,28*28],name='X2')
y = tf.placeholder(tf.float32,[None],name='y')
##############################################################################
# 
#W = tf.get_variable('W',initializer=tf.zeros([28*28,10]))
#b = tf.get_variable('b',initializer = tf.zeros([10]))

W = tf.get_variable('W',shape = (28*28,10),initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b',initializer = tf.zeros([10]))

logits11 = tf.matmul(X1,W) + b
logits1 = tf.nn.l2_normalize(logits11,axis=1)


logits22 = tf.matmul(X2,W) + b
logits2 = tf.nn.l2_normalize(logits22,axis=1)

m = .4

diff = logits1 - logits2 + 1e-16

# TODO: replace with diff
dist = tf.norm((diff),axis=1)

Ls = dist**2

#z = tf.zeros_like(dist)
z = tf.ones_like(dist)


Ld = tf.maximum(z,m - dist)**2

loss = (1-y)*Ls + y*Ld

diff_margin = m - dist
y_preds = tf.greater(0.0,m - dist)
correct_preds = tf.equal(tf.cast(y_preds,tf.float32),y)
n_correct = tf.reduce_sum(tf.cast(correct_preds,tf.float32))


reduced_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdagradOptimizer(1e-3).minimize(reduced_loss)
sess.run(tf.initialize_all_variables())

#for var in tf.trainable_variables():
#    tf.summary.histogram(var.name, var)
#merged_summary = tf.summary.merge_all()

inds_test1 = inds_test.copy()
inds_test2 = inds_test.copy()

np.random.shuffle(inds_test1)
np.random.shuffle(inds_test2)

for e in range(n_epoch):

    # TRAIN 
    inds_epoch1 = inds_train.copy()
    inds_epoch2 = inds_train.copy()
    
    np.random.shuffle(inds_epoch1)
    np.random.shuffle(inds_epoch2)
    
    total_loss = 0

    
    for i in range(n_batch_train):
    #for i in range(2):        
        
        inds_batch1 = inds_epoch1[i*batch_size_train:(i+1)*batch_size_train]
        inds_batch2 = inds_epoch2[i*batch_size_train:(i+1)*batch_size_train]
        
        X_batch1 = X_train[inds_batch1]
        y_batch1 = y_train[inds_batch1]

        X_batch2 = X_train[inds_batch2]
        y_batch2 = y_train[inds_batch2]
        
        y_batch = np.abs(y_batch1 - y_batch2)
        y_batch[y_batch>0] = 1
        
        
        feed_dict = {X1: X_batch1, X2: X_batch2, y: y_batch}     
        
        sess.run(optimizer,feed_dict)
        loss_batch = sess.run(reduced_loss,feed_dict) 

        total_loss += loss_batch
        
    print(total_loss)
    


    total_tp = 0
    total_tn = 0        
    total_fp = 0
    total_fn = 0
    
    total_correct = 0
    for i in range(n_batch_test):
        
        inds_batch1 = inds_test1[i*batch_size_test:(i+1)*batch_size_test]
        inds_batch2 = inds_test2[i*batch_size_test:(i+1)*batch_size_test]
        
        X_batch1 = X_test[inds_batch1]
        y_batch1 = y_test[inds_batch1]

        X_batch2 = X_test[inds_batch2]
        y_batch2 = y_test[inds_batch2]
        
        y_batch = np.abs(y_batch1 - y_batch2)
        y_batch[y_batch>0] = 1
        
        feed_dict = {X1: X_batch1, X2: X_batch2, y: y_batch} 
        total_correct += sess.run(n_correct,feed_dict)        
        preds = sess.run(y_preds,feed_dict)    
        preds = preds.astype(int)
        true_minus_pred = y_batch - preds
        
        n_false_negative = len(np.where(true_minus_pred==-1)[0])        
        n_false_positive = len(np.where(true_minus_pred==1)[0])        
        true_plus_pred = y_batch + preds
        n_true_negative = len(np.where(true_plus_pred==2)[0])        
        n_true_positive = batch_size_test - n_false_negative - n_false_positive - n_true_negative        

#        print(y_batch1)
#        print(y_batch2)
#        print(y_batch)
#        print(preds)
#        #print(preds.astype(int))
#        print(n_false_negative)
#        print("="*20)        
        
        total_tp += n_true_positive
        total_tn += n_true_negative        
        total_fp += n_false_positive
        total_fn += n_false_negative

        


    n_test_local =  n_batch_test*batch_size_test
    
    print('tp',total_tp/n_test_local,'tn',total_tn/n_test_local)
    print('fp',total_fp/n_test_local,'fn',total_fn/n_test_local)
    print(total_loss,total_correct/n_test_local)    
    print('='*batch_size_test*2)
        
        