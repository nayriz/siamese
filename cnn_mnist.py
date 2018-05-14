#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:51:50 2018

@author: john
"""

import numpy as np
import tensorflow as tf


X_train = np.load('/media/john/siamese/MNIST/X_train_MNIST.npy')
y_train = np.load('/media/john/siamese/MNIST/y_train_MNIST.npy')
X_test = np.load('/media/john/siamese/MNIST/X_test_MNIST.npy')
y_test = np.load('/media/john/siamese/MNIST/y_test_MNIST.npy')
n_train = X_train.shape[0]
n_test = X_test.shape[0]

X_train = np.reshape(X_train,(n_train,28*28))
X_test = np.reshape(X_test,(n_test,28*28))

inds_train = range(n_train)
inds_test = range(n_test)

batch_size_train = 128*2
batch_size_test = 64

n_batch_train = int(n_train/batch_size_train)
n_batch_test = int(n_test/batch_size_test)

inds_train = list(range(n_train))
inds_test = list(range(n_test))
n_epoch = 1
###############################################################################
#tf.reset_default_graph()
sess = tf.Session()

def model(X):
    
#    n_filter1 = 16
#    conv1 = tf.layers.conv2d(X,n_filter1,3,padding='SAME')    
#    conv1_relu = tf.nn.relu(conv1)
#    n_filter2 = 16    
#    n_end_features = 10
#    W = tf.Variable(tf.zeros([28*28*n_filter2,n_end_features]))    
#    conv2 = tf.layers.conv2d(conv1_relu,n_filter2,3,padding = 'SAME')
#    X2 = tf.reshape(conv2,[-1,28*28*n_filter2])
#    logits = tf.matmul(X2,W)
    
#    W = tf.Variable(tf.zeros([28*28,10]),dtype = tf.float32)
#    b = tf.Variable(tf.zeros([10]),dtype = tf.float32)
#    logits = tf.matmul(X,W) + b    
            
    return logits

#X1 = tf.placeholder(tf.float32,[None,28,28,1])
#X2 = tf.placeholder(tf.float32,[None,28,28,1])
X1 = tf.placeholder(tf.float32,[None,28*28])
X2 = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None])
##############################################################################

logits1 = model(X1)
logits2 = model(X2)
m = 1e6

#dist = tf.norm((logits1 - logits2),axis=1)
#Ls = dist**2
#z = tf.zeros_like(dist)
#Ld = tf.maximum(z,m - dist)**2
#loss = (1-y)*Ls + y*Ld

###########
dist = tf.reduce_sum((logits1 - logits2)**2,axis=1)
Ls = dist
z = tf.zeros_like(dist)
Ld = tf.maximum(z,m - dist)**2
loss = (1-y)*Ls + y*Ld
###########################

reduced_loss = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(1e0).minimize(reduced_loss)


sess.run(tf.initialize_all_variables())

for e in range(n_epoch):

    # TRAIN 
    inds_epoch1 = inds_train.copy()
    inds_epoch2 = inds_train.copy()
    
    np.random.shuffle(inds_epoch1)
    np.random.shuffle(inds_epoch2)
    
    total_loss = 0
    
    for i in range(n_batch_train):
        
        inds_batch1 = inds_epoch1[i*batch_size_train:(i+1)*batch_size_train]
        inds_batch2 = inds_epoch2[i*batch_size_train:(i+1)*batch_size_train]
        
        X_batch1 = X_train[inds_batch1]
        y_batch1 = y_train[inds_batch1]

        X_batch2 = X_train[inds_batch2]
        y_batch2 = y_train[inds_batch2]
        
        y_batch = np.abs(y_batch1 - y_batch2)
        y_batch[y_batch>0] = 1
        
        
        feed_dict = {X1: X_batch1, X2: X_batch2, y: y_batch}     

        piu = sess.run(y,feed_dict)
#        print(piu)
#        STOP
    
        loss_batch = sess.run(reduced_loss,feed_dict)    
        print(loss_batch)
        
        sess.run(optimizer,feed_dict)

        total_loss += loss_batch
        
    print(total_loss)