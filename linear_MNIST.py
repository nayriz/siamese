#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:02:35 2018

@author: john
"""
import numpy as np
import tensorflow as tf

Xtrain = np.load('MNIST/X_train_MNIST.npy')
ytrain = np.load('MNIST/y_train_MNIST.npy')

Xtest = np.load('MNIST/X_test_MNIST.npy')
ytest = np.load('MNIST/y_test_MNIST.npy')



#############
Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],28*28))
Xtest = np.reshape(Xtest,(Xtest.shape[0],28*28))
batch_size = 16
n_train = Xtrain.shape[0]
n_test = Xtest.shape[0]

n_batch = int(n_train/batch_size)
n_batch_test = int(n_test/batch_size)
 
inds = list(range(n_train))
inds_test = list(range(n_test)) 
n_epoch = 10

tf.reset_default_graph()
session = tf.Session()


X = tf.placeholder(tf.float32,[None, 28*28])
y = tf.placeholder(tf.int64,[None])
y_one_hot = tf.one_hot(y,10,axis=-1,dtype = tf.int64)
#W = tf.Variable(tf.zeros([28*28,10]))
W = tf.get_variable('W',initializer=tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))
logits = tf.matmul(X,W) + b
y_preds = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot,logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(.5).minimize(loss)
session.run(tf.initialize_all_variables())

y_preds_class = tf.argmax(y_preds,axis = 1)

correct_preds = tf.equal(y_preds_class,y)
accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))

for e in range(n_epoch):

    inds_epoch = inds.copy()
    np.random.shuffle(inds_epoch)
    
    total_loss = 0
    for i in range(n_batch):
        
        inds_batch = inds_epoch[i*batch_size:(i+1)*batch_size]
    
        X_batch = Xtrain[inds_batch]
        y_batch = ytrain[inds_batch]    
        
        feed_dict = {X: X_batch, y: y_batch}
        _ , loss_batch = session.run([optimizer,loss],feed_dict = feed_dict)
    
        total_loss += loss_batch
    
    print(total_loss)
    #print(total_loss)
    
total_correct = 0    
for i in range(n_batch_test):
    
    inds_batch = inds_test[i*batch_size:(i+1)*batch_size]

    X_batch = Xtest[inds_batch]
    y_batch = ytest[inds_batch]    
    feed_dict = {X: X_batch, y: y_batch}
    total_correct += session.run(accuracy,feed_dict = feed_dict)

acc = total_correct/(n_batch_test*batch_size)
print(acc)