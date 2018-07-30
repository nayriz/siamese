#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:12:25 2018

@author: john
"""

import numpy as np
import tensorflow as tf

X_train = np.load('data_google/X_train_small.npy')
y_train = np.load('data_google/y_train_small.npy')


n_class = np.max(y_train) + 1

X_train = np.expand_dims(X_train,axis=2)

X_test = np.load('data_google/X_test_small.npy')
X_test = np.expand_dims(X_test,axis=2)

y_test = np.load('data_google/y_test_small.npy')

n_train = X_train.shape[0]
n_test = X_test.shape[0]

batch_size_train = 32
batch_size_test = 128

n_batch_train = int(np.ceil(n_train/batch_size_train))
n_batch_test = int(np.ceil(n_test/batch_size_test))

inds_train = list(range(n_train))
inds_test = list(range(n_test))

n_epoch = 100

tf.reset_default_graph()
sess = tf.Session()
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "talisol:7000")

def n_correct_preds(y_preds_class,labels):
    
    correct_i = tf.equal(y_preds_class,labels)
    
    n_correct = tf.reduce_sum(tf.cast(correct_i,tf.float32))
    
    return n_correct

X = tf.placeholder(tf.float32,shape=[None,16000,1],name='X')
labels = tf.placeholder(tf.int64,shape=[None],name = 'y')


filter1 = 16
kernel1 = 3
conv1 = tf.layers.conv1d(X,filter1,kernel1,activation = tf.nn.relu, padding = 'SAME',name='conv1')

filter2 = 16
kernel2 = 3
conv2 = tf.layers.conv1d(conv1,filter2,kernel2, padding = 'SAME',name = 'conv2')

Xf = tf.reshape(conv2,[-1,filter2*16000])

logits = tf.layers.dense(Xf,n_class)

labels_one_hot = tf.one_hot(labels,n_class,axis=-1)
loss_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels_one_hot)

loss = tf.reduce_mean(loss_i,name='loss')

optimizer = tf.train.AdamOptimizer(.0005).minimize(loss) 

y_preds = tf.nn.softmax(logits)
y_preds_class = tf.argmax(y_preds,axis=-1)
n_correct = n_correct_preds(y_preds_class,labels)    

##############################################################################
sess.run(tf.initialize_all_variables())

for e in range(n_epoch):

    # TRAIN 
    inds_epoch = inds_train.copy()
    np.random.shuffle(inds_epoch)
    
    total_loss = 0
    
    for i in range(n_batch_train):
        
        if i == n_batch_train-1:
            inds_batch = inds_epoch[i*batch_size_train:] 
        else :           
            inds_batch = inds_epoch[i*batch_size_train:(i+1)*batch_size_train]            
        X_batch = X_train[inds_batch]
        y_batch = y_train[inds_batch]
        
        feed_dict = {X: X_batch, labels: y_batch}
        
        sess.run(optimizer,feed_dict)
        total_loss += sess.run(loss,feed_dict)    
      


    # TEST    
    total_correct = 0
    total_samples = 0          
    for i in range(n_batch_test):

        if i == n_batch_train-1:
            inds_batch = inds_test[i*batch_size_test:]
        else :    
            inds_batch = inds_test[i*batch_size_test:(i+1)*batch_size_test] 
            
        X_batch = X_test[inds_batch]
        y_batch = y_test[inds_batch]
        
        feed_dict = {X: X_batch, labels: y_batch}
        
        total_correct += sess.run(n_correct,feed_dict)
        total_samples += len(inds_batch)
        
    print(total_loss,total_correct,total_samples,total_correct/(n_batch_test*batch_size_test))
    
    # TRAINING ACCURACY
#    total_correct = 0
#    total_samples = 0    
#    for i in range(n_batch_train):
#        
#        if i == n_batch_train-1:
#            inds_batch = inds_epoch[i*batch_size_train:] 
#        else :           
#            inds_batch = inds_epoch[i*batch_size_train:(i+1)*batch_size_train]            
#        X_batch = X_train[inds_batch]
#        y_batch = y_train[inds_batch]
#        
#        feed_dict = {X: X_batch, labels: y_batch}
#        
#        total_correct += sess.run(n_correct,feed_dict)
#        total_samples += len(inds_batch)
#
#    print(total_loss,total_correct,total_samples,total_correct/(n_batch_train*batch_size_train))    