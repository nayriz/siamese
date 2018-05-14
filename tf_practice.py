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

batch_size_train = 32
batch_size_test = 64

n_batch_train = int(n_train/batch_size_train)
n_batch_test = int(n_test/batch_size_test)

inds_train = list(range(n_train))
inds_test = list(range(n_test))
n_epoch = 100
###############################################################################
#tf.reset_default_graph()
sess = tf.Session()


#X = tf.placeholder(tf.float32,[None,28*28])
X = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.int64,[None])

is_training = tf.placeholder(tf.bool)
y_one_hot = tf.one_hot(y,10,axis = -1)


# MODEL 1
#W = tf.Variable(tf.zeros([28*28,10]),dtype = tf.float32)
#b = tf.Variable(tf.zeros([10]),dtype = tf.float32)
#logits = tf.matmul(X,W) + b


##############################################################################
# MODEL 2

def model2(is_training):
    W1 = tf.Variable(tf.zeros([28*28,10]),dtype = tf.float32)
    b1 = tf.Variable(tf.zeros([10]),dtype = tf.float32)
    X2 = tf.matmul(X,W1) + b1
    X2relu = tf.nn.relu(X2)
    #X2bn = tf.layers.batch_normalization(X2relu,training = is_training)
    
    W2 = tf.Variable(tf.zeros([10,10]),dtype = tf.float32)
    b2 = tf.Variable(tf.zeros([10]), dtype = tf.float32)
    logits = tf.matmul(X2relu,W2) + b2
    
    return logits

def model3(X):
    n_filter1 = 16
    X1 = tf.layers.conv2d(X,n_filter1,3,padding='SAME')

    
    X1relu = tf.nn.relu(X1)
    n_filter2 = 16
    W = tf.Variable(tf.zeros([28*28*n_filter2,10]))
    X2conv = tf.layers.conv2d(X1relu,n_filter2,3,padding = 'SAME')
    X2 = tf.reshape(X2conv,[-1,28*28*n_filter2])
    logits = tf.matmul(X2,W)
    
    #return logits
    
    return logits
    
##############################################################################
#logits = model2(is_training)
logits = model3(X)
y_preds = tf.nn.softmax(logits)
y_preds_class = tf.argmax(y_preds,axis = 1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_one_hot,logits = logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(.0005).minimize(loss)
correct_preds = tf.equal(y,y_preds_class)
n_correct = tf.reduce_sum(tf.cast(correct_preds,tf.int16))


sess.run(tf.initialize_all_variables())

for e in range(n_epoch):

    # TRAIN 
    inds_epoch = inds_train.copy()
    np.random.shuffle(inds_epoch)
    
    total_loss = 0
    
    for i in range(n_batch_train):
        
        inds_batch = inds_epoch[i*batch_size_train:(i+1)*batch_size_train]
        X_batch = X_train[inds_batch]
        y_batch = y_train[inds_batch]
        
        feed_dict = {X: X_batch, y: y_batch, is_training:True}
        
#        piu = sess.run(logits,feed_dict)
#        STOP
        
        sess.run(optimizer,feed_dict)
        total_loss += sess.run(loss,feed_dict)    
        
    # TEST
    
    total_correct = 0
    for i in range(n_batch_test):
        
        inds_batch = inds_test[i*batch_size_test:(i+1)*batch_size_test]
        X_batch = X_test[inds_batch]
        y_batch = y_test[inds_batch]
        
        feed_dict = {X: X_batch, y: y_batch, is_training:False}
                
        total_correct += sess.run(n_correct,feed_dict)


    print(total_loss,total_correct/(n_batch_test*batch_size_test))

weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

