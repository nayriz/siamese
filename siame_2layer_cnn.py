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

#X_train = np.reshape(X_train,(n_train,28*28))
#X_test = np.reshape(X_test,(n_test,28*28))

inds_train = range(n_train)
inds_test = range(n_test)

batch_size_train = 128*2
batch_size_test = 8

n_batch_train = int(n_train/batch_size_train)
n_batch_test = int(n_test/batch_size_test)

inds_train = list(range(n_train))
inds_test = list(range(n_test))
n_epoch = 100
###############################################################################
tf.reset_default_graph()
sess = tf.Session()
n_end_features = 128    
#W = tf.get_variable('W',initializer=tf.zeros([28*28,n_end_features]))  
W = tf.get_variable('W', shape = [16*28*28,n_end_features], initializer = tf.contrib.layers.xavier_initializer())  
b = tf.get_variable('b',initializer=tf.zeros([n_end_features]),dtype = tf.float32)

def model(X):
    
    n_filter1 = 16
    conv1 = tf.layers.conv2d(X,n_filter1,3,padding='SAME')    
    conv1_relu = tf.nn.relu(conv1)
    n_filter2 = 16    
    n_end_features = 10
    #W = tf.Variable(tf.zeros([28*28*n_filter2,n_end_features]))    
    conv2 = tf.layers.conv2d(conv1_relu,n_filter2,3,padding = 'SAME')
    X2 = tf.reshape(conv2,[-1,28*28*n_filter2])
    logits = tf.matmul(X2,W)

#    n_end_features = 128    
##    W = tf.Variable(tf.zeros([28*28,n_end_features]),dtype = tf.float32)
##    W = tf.get_variable('W')    
#    b = tf.Variable(tf.zeros([n_end_features]),dtype = tf.float32)
#    logits = tf.matmul(X,W) + b    
            
    return logits

X1 = tf.placeholder(tf.float32,[None,28,28,1])
X2 = tf.placeholder(tf.float32,[None,28,28,1])
#X1 = tf.placeholder(tf.float32,[None,28*28])
#X2 = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None])
##############################################################################

logits1 = model(X1)
logits2 = model(X2)
m = 10
diff = logits1 - logits2

dist = tf.norm((logits1 - logits2),axis=1)

Ls = dist**2

z = tf.zeros_like(dist)

Ld = tf.maximum(z,m - dist)**2

loss = (1-y)*Ls + y*Ld

diff_margin = m - dist
y_preds = tf.greater(0.0,m - dist)
correct_preds = tf.equal(tf.cast(y_preds,tf.float32),y)
n_correct = tf.reduce_sum(tf.cast(correct_preds,tf.float32))


reduced_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(1e-3).minimize(reduced_loss)


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
        #print(loss_batch)
        
        sess.run(optimizer,feed_dict)

        total_loss += loss_batch
        
    #print(total_loss)
    
    inds_test1 = inds_test.copy()
    inds_test2 = inds_test.copy()
    
    np.random.shuffle(inds_test1)
    np.random.shuffle(inds_test2)
    
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
        
        print(y_batch1)
        print(y_batch2)

        feed_dict = {X1: X_batch1, X2: X_batch2, y: y_batch}   
        preds = sess.run(y_preds,feed_dict)
        print(y_batch)
        print(preds)
        total_correct += sess.run(n_correct,feed_dict)
        
        STOP


    print(total_loss,total_correct/(n_batch_test*batch_size_test))
    
    acc = total_correct/(n_batch_test*batch_size_test)
        