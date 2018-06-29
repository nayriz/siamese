#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:51:50 2018

@author: john
"""

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

X_train = np.load('MNIST/X_train_MNIST.npy')
y_train = np.load('MNIST/y_train_MNIST.npy')
X_test = np.load('MNIST/X_test_MNIST.npy')
y_test = np.load('MNIST/y_test_MNIST.npy')

n_train = X_train.shape[0]
n_test = X_test.shape[0]

classes = list(range(10))
np.random.shuffle(classes)
train_classes = classes[:8]
test_classes = classes[8:10]
inds_train = {}
INDS_anchor = []
inds_positive = {}
INDS_positive = []


# initialize the total number of training samples
N_train = 0

for i in train_classes:
    
    # find the indices each class in the training set
    ind_i = list(np.where(y_train == i)[0])

    # save the indices for each class
    inds_train[i] = ind_i
    N_train += len(ind_i)
    
# initialize the dictionary of indices of all negative 
# samples corresonding to a class
inds_all_negative = {}
ind_train = np.array(range(n_train))

for i in train_classes:       
    all_inds_diff_i = ind_train[ np.where(y_train != i)[0]]
    inds_all_negative[i] = all_inds_diff_i

# TODO: this might need to be changed to be the same as the training examples
list_test = []
for i in test_classes:    
    list_test += np.ndarray.tolist(np.where(y_test == i)[0])
    
###############################################################################

batch_size_train = 16
batch_size_test = 16

n_batch_train = int(n_train/batch_size_train)
n_batch_test = int(n_test/batch_size_test)

all_inds_train = list(range(n_train))
inds_test = list(range(n_test))

inds_test1 = inds_test.copy()
inds_test2 = inds_test.copy()

np.random.shuffle(inds_test1)
np.random.shuffle(inds_test2)
n_epoch = 1000
###############################################################################
tf.reset_default_graph()
sess = tf.Session()



X1 = tf.placeholder(tf.float32,[None,28,28,1],name = 'X1')
X2 = tf.placeholder(tf.float32,[None,28,28,1],name = 'X2')
X3 = tf.placeholder(tf.float32,[None,28,28,1],name = 'X3')
#X1 = tf.placeholder(tf.float32,[None,28*28])
#X2 = tf.placeholder(tf.float32,[None,28*28])
#y = tf.placeholder(tf.float32,[None])
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
    
    logits1 = tf.layers.dense(XfX1,n_final_features)    
    
    logits_anchor = tf.nn.l2_normalize(logits1,axis=1)

with tf.variable_scope('conv',reuse=True):
    
    conv1X2 = tf.layers.conv2d(X2,n_filter1,kernel1,activation = tf.nn.relu,reuse=None,padding='SAME',name='conv1')    
    conv2X2 = tf.layers.conv2d(conv1X2,n_filter2,kernel2,padding = 'SAME',name='conv2') 


    XfX2 = tf.reshape(conv2X2,[-1,n_filter_end*28*28])

    logits2 = tf.layers.dense(XfX2,n_final_features)    
    
    logits_positive = tf.nn.l2_normalize(logits2,axis=1)
    
with tf.variable_scope('conv',reuse=True):
    
    conv1X3 = tf.layers.conv2d(X3,n_filter1,kernel1,activation = tf.nn.relu,reuse=None,padding='SAME',name='conv1')    
    conv2X3 = tf.layers.conv2d(conv1X3,n_filter2,kernel2,padding = 'SAME',name='conv2') 


    XfX3 = tf.reshape(conv2X3,[-1,n_filter_end*28*28])

    logits3 = tf.layers.dense(XfX3,n_final_features)    
    
    logits_negative = tf.nn.l2_normalize(logits3,axis=1)    

##############################################################################

margin = .1

d_positive = tf.norm(logits_anchor - logits_positive + 1e-16,axis=1)**2
d_negative = tf.norm(logits_anchor - logits_negative,axis=1)**2


#d_positive = tf.reduce_sum(logits_anchor - logits_positive + 1e-16,axis=1)**2
#d_negative = tf.reduce_sum(logits_anchor - logits_negative,axis=1)**2

l0 = d_positive - d_negative + margin

#z = tf.zeros_like(l0)

with tf.variable_scope('loss'):
    loss = tf.maximum(0.0,l0)


#diff_margin = m - dist
#y_preds = tf.greater(0.0,m - dist)
#correct_preds = tf.equal(tf.cast(y_preds,tf.float32),y)
#n_correct = tf.reduce_sum(tf.cast(correct_preds,tf.float32))


reduced_loss = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(1e-1).minimize(reduced_loss)
#optimizer = tf.train.AdamOptimizer(1e-3).minimize(reduced_loss)

#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "talisol:7000")
sess.run(tf.initialize_all_variables())

INDS = list(range(N_train))

for e in range(3):
    np.random.shuffle(INDS)
    # TRAIN 
    inds_anchor = []
    inds_positive = []
    inds_negative = []
    
    for i in train_classes:
        
        ind_i = inds_train[i] 
        ind_i0 = ind_i.copy()
        np.random.shuffle(ind_i)
        np.random.shuffle(ind_i0)
        
        ind_neg = list(np.random.choice(inds_all_negative[i],size = len(inds_train[i]),replace = False))
        inds_anchor += ind_i                
        inds_positive += ind_i0
        inds_negative += ind_neg

    
    inds_anchor = np.array(inds_anchor)
    inds_positive = np.array(inds_positive)
    inds_negative = np.array(inds_negative)
    
    inds_anchor = inds_anchor[INDS]
    inds_positive = inds_positive[INDS]
    inds_negative = inds_negative[INDS]        
            
    total_loss = 0    
    for i in range(n_batch_train):
        
        inds_anchor = inds_anchor[i*batch_size_train:(i+1)*batch_size_train]
        inds_positive = inds_positive[i*batch_size_train:(i+1)*batch_size_train]
        inds_negative = inds_negative[i*batch_size_train:(i+1)*batch_size_train]
        
        X_anchor = X_train[inds_anchor]
        y_anchor = y_train[inds_anchor]

        X_positive = X_train[inds_positive]
        y_positive = y_train[inds_positive]
        
        X_negative = X_train[inds_negative]
        y_negative = y_train[inds_negative]        
                
#        y_batch = np.abs(y_batch1 - y_batch2)
#        y_batch[y_batch>0] = 1
        
        y_batch = []
        
        feed_dict = {X1: X_anchor, X2: X_positive, X3: X_negative}     

#        l_anchor = sess.run(logits_anchor,feed_dict) 
#        l_positive = sess.run(logits_positive,feed_dict) 
#        l_negative = sess.run(logits_negative,feed_dict) 
#
        l_anchor, l_positive, l_negative, d_pos, d_neg, l  = sess.run([logits_anchor,logits_positive,logits_negative,d_positive, d_negative, loss],feed_dict) 

#        print(l_anchor)
#        print(l_positive)
#        print(l_negative)
#        print(d_neg)
#        print(d_pos)
#        print(l,len(l))
#        
#        print(len(l_anchor))
#        print(len(l_positive))
#        print(len(l_negative))
#        print(len(d_neg))
#        print(len(d_pos))        
#        
#        if len(d_neg) == 0:
#            STOP
      


        
#        loss_batch0 = sess.run(reduced_loss,feed_dict)    
#        print(loss_batch0)
#        STOP
        loss_batch = sess.run(reduced_loss,feed_dict)            
        total_loss += loss_batch
        print(total_loss)

        sess.run(optimizer,feed_dict)
        
     
    
#        print(total_loss)
#        STOP
    
#    inds_test1 = inds_test.copy()
#    inds_test2 = inds_test.copy()
    
#    np.random.shuffle(inds_test1)
#    np.random.shuffle(inds_test2)
#
#    total_tp = 0
#    total_tn = 0        
#    total_fp = 0
#    total_fn = 0
#    
#    total_correct = 0
#    for i in range(n_batch_test):
#        
#        inds_batch1 = inds_test1[i*batch_size_test:(i+1)*batch_size_test]
#        inds_batch2 = inds_test2[i*batch_size_test:(i+1)*batch_size_test]
#        
#        X_batch1 = X_test[inds_batch1]
#        y_batch1 = y_test[inds_batch1]
#
#        X_batch2 = X_test[inds_batch2]
#        y_batch2 = y_test[inds_batch2]
#        
#        y_batch = np.abs(y_batch1 - y_batch2)
#        y_batch[y_batch>0] = 1
#        
#        feed_dict = {X1: X_batch1, X2: X_batch2, y: y_batch} 
#        total_correct += sess.run(n_correct,feed_dict)        
#        preds = sess.run(y_preds,feed_dict)    
#        preds = preds.astype(int)
#        true_minus_pred = y_batch - preds
#        
#        n_false_negative = len(np.where(true_minus_pred==-1)[0])        
#        n_false_positive = len(np.where(true_minus_pred==1)[0])        
#        true_plus_pred = y_batch + preds
#        n_true_negative = len(np.where(true_plus_pred==2)[0])        
#        n_true_positive = batch_size_test - n_false_negative - n_false_positive - n_true_negative        
#
##        print(y_batch1)
##        print(y_batch2)
##        print(y_batch)
##        print(preds)
##        #print(preds.astype(int))
##        print(n_false_negative)
##        print("="*20)        
#        
#        total_tp += n_true_positive
#        total_tn += n_true_negative        
#        total_fp += n_false_positive
#        total_fn += n_false_negative
#
#        
#
#
#    n_test_local =  n_batch_test*batch_size_test
#    
#    print('tp',total_tp/n_test_local,'tn',total_tn/n_test_local)
#    print('fp',total_fp/n_test_local,'fn',total_fn/n_test_local)
#    print(total_loss,total_correct/n_test_local)    
#    print('='*batch_size_test*2)
#        
#        