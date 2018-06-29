#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:51:50 2018

@author: john
"""

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import math
X_train = np.load('MNIST/X_train_MNIST.npy')
y_train = np.load('MNIST/y_train_MNIST.npy')
X_test = np.load('MNIST/X_test_MNIST.npy')
y_test = np.load('MNIST/y_test_MNIST.npy')

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

np.random.shuffle(list_train)
n_train = len(list_train)
list_train1 = list_train[:int(n_train/2)]
list_train2 = list_train[int(n_train/2):int(n_train/2)*2]
len(list_train1)
len(list_train2)
print(len(list_train1),len(list_train2))
print(n_train)

X_train1 = X_train[list_train1]
y_train1 = y_train[list_train1]

X_train2 = X_train[list_train2]
y_train2 = y_train[list_train2]

X_test = X_test[list_test]
y_test = y_test[list_test]

n_train = X_train.shape[0]
n_test = X_test.shape[0]


#X_train = np.reshape(X_train,(n_train,28*28))
#X_test = np.reshape(X_test,(n_test,28*28))

inds_train = range(n_train)
inds_test = range(n_test)

batch_size_train = 32
batch_size_test = 16

n_batch_train = int(n_train/batch_size_train)
n_batch_test = int(n_test/batch_size_test)

inds_train = list(range(len(list_train)))
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



X1 = tf.placeholder(tf.float32,[None,28,28,1],name = 'X1')
X2 = tf.placeholder(tf.float32,[None,28,28,1],name = 'X2')
#X1 = tf.placeholder(tf.float32,[None,28*28])
#X2 = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None],name = 'y')

##############################################################################
n_filter1 = 16
kernel1 = 3
n_filter2 = 16     
kernel2 = 3
n_filter_end = n_filter2
n_final_features = 16
    
with tf.variable_scope('conv'):
    conv1X1 = tf.layers.conv2d(X1,n_filter1,kernel1,activation = tf.nn.relu,reuse=None,padding='SAME',name='conv1')    
    conv2X1 = tf.layers.conv2d(conv1X1,n_filter2,kernel2,padding = 'SAME',name='conv2') 


    XfX1 = tf.reshape(conv2X1,[-1,n_filter_end*28*28])
    
    logits11 = tf.layers.dense(XfX1,n_final_features)    
    
    logits1 = tf.nn.l2_normalize(logits11,axis=1)

with tf.variable_scope('conv',reuse=True):
    conv1X2 = tf.layers.conv2d(X2,n_filter1,kernel1,activation = tf.nn.relu,reuse=None,padding='SAME',name='conv1')    
    conv2X2 = tf.layers.conv2d(conv1X2,n_filter2,kernel2,padding = 'SAME',name='conv2') 


    XfX2 = tf.reshape(conv2X2,[-1,n_filter_end*28*28])

    logits22 = tf.layers.dense(XfX2,n_final_features)    
    
    logits2 = tf.nn.l2_normalize(logits22,axis=1)

##############################################################################

m = .1

diff = logits1 - logits2 + 1e-16

# TODO: replace with diff
dist = tf.norm((diff),axis=1)

Ls = dist**2

z = tf.zeros_like(dist)

Ld = tf.maximum(z,m - dist)**2

with tf.variable_scope('loss'):
    loss = (1-y)*Ls + y*Ld

diff_margin = m - dist
y_preds = tf.greater(0.0,m - dist)
correct_preds = tf.equal(tf.cast(y_preds,tf.float32),y)
n_correct = tf.reduce_sum(tf.cast(correct_preds,tf.float32))


reduced_loss = tf.reduce_mean(loss,name = 'reduced_loss')
#optimizer = tf.train.AdamOptimizer(1e-3).minimize(reduced_loss)
optimizer = tf.train.AdagradOptimizer(1e-1).minimize(reduced_loss)
sess.run(tf.initialize_all_variables())
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "talisol:7000")
#for var in tf.trainable_variables():
#    tf.summary.histogram(var.name, var)
#merged_summary = tf.summary.merge_all()



for e in range(n_epoch):

    print('\nepoch',e)
    print('='*20)
    # TRAIN 
    inds_train1 = inds_train.copy()
    inds_train2 = inds_train.copy()
    
    np.random.shuffle(inds_train1)
    np.random.shuffle(inds_train2)
    
    total_loss = 0

    
    for i in range(n_batch_train):
        
        inds_batch1 = inds_train1[i*batch_size_train:(i+1)*batch_size_train]
        inds_batch2 = inds_train2[i*batch_size_train:(i+1)*batch_size_train]
        
        X_batch1 = X_train[inds_batch1]
        y_batch1 = y_train[inds_batch1]

        X_batch2 = X_train[inds_batch2]
        y_batch2 = y_train[inds_batch2]
        
        y_batch = np.abs(y_batch1 - y_batch2)
        y_batch[y_batch>0] = 1
        
        
        feed_dict = {X1: X_batch1, X2: X_batch2, y: y_batch}     
        sess.run(optimizer,feed_dict)
        #loss_batch = sess.run(reduced_loss,feed_dict)    
        
        #total_loss += loss_batch
        #print(i,loss_batch)
        #print(i)
        
#        if math.isnan(loss_batch):
#            STOP
    

    inds_test1 = inds_test.copy()
    inds_test2 = inds_test.copy()
    
    np.random.shuffle(inds_test1)
    np.random.shuffle(inds_test2)
    
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