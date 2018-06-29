#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 09:53:52 2018

@author: john
"""

import numpy as np
import tensorflow as tf
import os
import shutil
from tensorflow.python import debug as tf_debug

X_train = np.load('MNIST/X_train_MNIST.npy')
y_train = np.load('MNIST/y_train_MNIST.npy')
X_test = np.load('MNIST/X_test_MNIST.npy')
y_test = np.load('MNIST/y_test_MNIST.npy')
n_train = X_train.shape[0]
n_test = X_test.shape[0]

batch_size_train = 32
batch_size_test = 128

n_batch_train = int(n_train/batch_size_train)
n_batch_test = int(n_test/batch_size_test)

inds_train = list(range(n_train))
inds_test = list(range(n_test))

n_epoch = 5

tf.reset_default_graph()
sess = tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "talisol:7000")

def cnn(X):
    
    filter1 = 16
    kernel1 = 3
    #X0 = tf.reshape(X,[-1,28,28,1])
    X0 = X

    conv1 = tf.layers.conv2d(X0,filter1,kernel1,activation = tf.nn.relu, padding = 'SAME')


    
    filter2 = 16
    kernel2 = 3

    conv2 = tf.layers.conv2d(conv1,filter2,kernel2, padding = 'SAME')

 

    Xf = tf.reshape(conv2,[-1,filter2*28*28])
    
    logits = tf.layers.dense(Xf,10)
    
    return logits


def loss_func(logits,labels):
    
    labels_one_hot = tf.one_hot(labels,10,axis=-1)
    loss_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels_one_hot)
    
    loss = tf.reduce_mean(loss_i)
    
    return loss

def n_correct_preds(y_preds_class,labels):
    
    correct_i = tf.equal(y_preds_class,labels)
    
    n_correct = tf.reduce_sum(tf.cast(correct_i,tf.float32))
    
    return n_correct
   

X = tf.placeholder(tf.float32,shape=[None,28,28,1],name='X')
labels = tf.placeholder(tf.int64,shape=[None],name='labels')
logits = cnn(X)
loss = loss_func(logits,labels)
#tf.summary.scalar('loss')
optimizer = tf.train.GradientDescentOptimizer(.0005).minimize(loss) 

logdir = './logs/cnn_mnist/'
if os.path.isdir(logdir):
    shutil.rmtree(logdir)

writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)
y_preds = tf.nn.softmax(logits)
y_preds_class = tf.argmax(y_preds,axis=-1)
n_correct = n_correct_preds(y_preds_class,labels)    
merged_summaries = tf.summary.merge_all()
##############################################################################
sess.run(tf.global_variables_initializer())

for e in range(n_epoch):

    # TRAIN 
    inds_epoch = inds_train.copy()
    np.random.shuffle(inds_epoch)
    
    total_loss = 0
    
    for i in range(n_batch_train):
                
        inds_batch = inds_epoch[i*batch_size_train:(i+1)*batch_size_train]
        X_batch = X_train[inds_batch]
        y_batch = y_train[inds_batch]
        
        feed_dict = {X: X_batch, labels: y_batch}
        
        sess.run(optimizer,feed_dict)
        total_loss += sess.run(loss,feed_dict)    
        s = sess.run(merged_summaries,feed_dict) 
        writer.add_summary(s,i)
    # TEST    
    total_correct = 0
    for i in range(n_batch_test):

        inds_batch = inds_test[i*batch_size_test:(i+1)*batch_size_test]
        X_batch = X_test[inds_batch]
        y_batch = y_test[inds_batch]
        
        feed_dict = {X: X_batch, labels: y_batch}
        
        total_correct += sess.run(n_correct,feed_dict)

    print(total_loss,total_correct/(n_batch_test*batch_size_test))





    
    
    