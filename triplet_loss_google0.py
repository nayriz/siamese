#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:51:50 2018

@author: john
"""

import numpy as np
import tensorflow as tf
import json
import copy 
#X_train = np.load('data_google/X_train_small.npy')
#y_train = np.load('data_google/y_train_small.npy')



#n_select = int(len(X_train))
#
#list_select = list(range(n_select))
#np.random.shuffle(list_select)
#X_train = X_train[list_select]
#y_train = y_train[list_select]



###########################
with open('data_google/positive_train.json') as f:   
    positive_train = json.load(f)
f.close()

with open('data_google/negative_train.json') as f:   
    negative_train = json.load(f)
f.close()


X_train = np.load('data_google/X_train_small.npy')
X_train = np.expand_dims(X_train,axis=2)

y_train = np.load('data_google/y_train_small.npy')

X_test = np.load('data_google/X_test_small.npy')
X_test = np.expand_dims(X_test,axis=2)

y_test = np.load('data_google/y_test_small.npy')

###############################################################################
#X_train = np.load('data_google/X_.npy')
#X_train = np.expand_dims(X_train,axis=2)
#
#y_train = np.load('data_google/y.npy')
#
#X_test = np.load('data_google/X.npy')
#X_test = np.expand_dims(X_test,axis=2)
#
#y_test = np.load('data_google/y.npy')

###############################################################################
n_classes = np.max(y_train)
classes = list(range(n_classes))

np.random.shuffle(classes)

train_classes = classes[:n_classes-2]
train_classes = classes
test_classes = classes[n_classes-2:]

###############################################################################
inds_positive = {}
for i in train_classes:
    
    # find the indices each class in the training set
    ind_i = list(np.where(y_train == i)[0])

    # save the indices for each class
    inds_positive[i] = ind_i

positive_train = inds_positive

inds_all_negative = {}
for i in train_classes:       
    
    l0 = list(np.where(y_train != i)[0])
    
    for j in test_classes:
        
        l = list(np.where(y_train != j)[0])
        
        l0 = list(set(l0) & set(l))

    inds_all_negative[i] = l0
    
negative_train =  inds_all_negative   
###############################################################################
    
    
list_test = []
for i in test_classes:    
    list_test += np.ndarray.tolist(np.where(y_test == i)[0])

print('comparing',test_classes[0],'and',test_classes[1])

X_test = X_test[list_test]
y_test = y_test[list_test]

n_test = X_test.shape[0]


batch_size_train = 32
batch_size_test = 32


n_batch_test = int(n_test/batch_size_test)

inds_test = list(range(n_test))

n_epoch = 1000
###############################################################################
tf.reset_default_graph()
sess = tf.Session()

X1 = tf.placeholder(tf.float32,shape=[None,16000,1],name='X1')
X2 = tf.placeholder(tf.float32,shape=[None,16000,1],name='X2')
X3 = tf.placeholder(tf.float32,shape=[None,16000,1],name='X3')

y = tf.placeholder(tf.float32,[None],name = 'y')


labels = tf.placeholder(tf.int64,shape=[None],name = 'y')

##############################################################################
filter1 = 16
kernel1 = 3

filter2 = 16
kernel2 = 3
n_class = 10

with tf.variable_scope('conv'):    

    conv11 = tf.layers.conv1d(X1,filter1,kernel1,activation = tf.nn.relu, padding = 'SAME',name='conv1') 
    conv21 = tf.layers.conv1d(conv11,filter2,kernel2, padding = 'SAME',name = 'conv2')
    Xf1 = tf.reshape(conv21,[-1,filter2*16000])
    
    logits1 = tf.layers.dense(Xf1,n_class)

    logits_anchor = tf.nn.l2_normalize(logits1,axis=1)

with tf.variable_scope('conv',reuse=True):
    
    conv12 = tf.layers.conv1d(X2,filter1,kernel1,activation = tf.nn.relu, padding = 'SAME',name='conv1') 
    conv22 = tf.layers.conv1d(conv12,filter2,kernel2, padding = 'SAME',name = 'conv2')
    Xf2 = tf.reshape(conv22,[-1,filter2*16000])
    
    logits2 = tf.layers.dense(Xf2,n_class)

    logits_positive = tf.nn.l2_normalize(logits2,axis=1)
    
with tf.variable_scope('conv',reuse=True):
    
    conv13 = tf.layers.conv1d(X3,filter1,kernel1,activation = tf.nn.relu, padding = 'SAME',name='conv1') 
    conv23 = tf.layers.conv1d(conv13,filter2,kernel2, padding = 'SAME',name = 'conv2')
    Xf3 = tf.reshape(conv23,[-1,filter2*16000])
    
    logits3 = tf.layers.dense(Xf3,n_class)

    logits_negative = tf.nn.l2_normalize(logits3,axis=1)
##############################################################################

m = .1
diff_ap = logits_anchor - logits_positive
dist_ap = tf.norm((diff_ap + 1e-16),axis=1)

diff_an = logits_anchor - logits_negative
dist_an = tf.norm((diff_an + 1e-16),axis=1)

z = tf.zeros_like(dist_ap)

with tf.variable_scope('loss'):
    loss = tf.maximum(z,dist_ap - dist_an + m)

###############################################################################
#m_test = .15 
m_test = tf.placeholder(tf.float32,shape=[],name='margin_test')
diff_margin = m_test - dist_ap
y_preds = tf.greater(0.0,diff_margin)
correct_preds = tf.equal(tf.cast(y_preds,tf.float32),y)
n_correct = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
###############################################################################


reduced_loss = tf.reduce_mean(loss,name = 'reduced_loss')
optimizer = tf.train.AdamOptimizer(1e-3).minimize(reduced_loss)
sess.run(tf.initialize_all_variables())

###############################################################################
inds_test1 = inds_test.copy()
inds_test2 = inds_test.copy()

np.random.shuffle(inds_test1)
np.random.shuffle(inds_test2)  


#n_batch_train = int(len(inds_train)/batch_size_train)


for e in range(n_epoch):
    m_test0 = .5
    inds_train_anchor = []
    inds_train_positive = []
    inds_train_negative = []
    
    for i in train_classes :

        y_epoch = positive_train[i]
        
        # find a way of excluding 5 and 7
        y_neg = negative_train[i] 
        np.random.shuffle(y_epoch)
        np.random.shuffle(y_neg)    
        n_y = len(y_epoch)
        
        r_divide = 2
        inds_train_anchor += y_epoch[:int(n_y/r_divide)]
        inds_train_positive += y_epoch[int(n_y/r_divide):r_divide*int(n_y/r_divide)]
        inds_train_negative += y_neg[:(r_divide-1)*int(n_y/r_divide)]        
        
  
    inds_epoch = list(range(len(inds_train_anchor)))
    np.random.shuffle(inds_epoch)
        
    inds_train_anchor = np.array(inds_train_anchor)    
    inds_train_positive = np.array(inds_train_positive)    
    inds_train_negative = np.array(inds_train_negative)    
    
    
    inds_train_anchor = inds_train_anchor[inds_epoch]
    inds_train_positive = inds_train_positive[inds_epoch]
    inds_train_negative = inds_train_negative[inds_epoch]
    
    n_train = len(inds_epoch)
    
    n_batch_train = int(len(inds_epoch)/batch_size_train)
        
    total_loss = 0
    total_diff = 0
    
    for i in range(n_batch_train):
        
        inds_batch1 = inds_train_anchor[i*batch_size_train:(i+1)*batch_size_train]
        inds_batch2 = inds_train_positive[i*batch_size_train:(i+1)*batch_size_train]
        inds_batch3 = inds_train_negative[i*batch_size_train:(i+1)*batch_size_train]        
        
        X_batch1 = X_train[inds_batch1]
        y_batch1 = y_train[inds_batch1]

        X_batch2 = X_train[inds_batch2]
        y_batch2 = y_train[inds_batch2]

        X_batch3 = X_train[inds_batch3]
        y_batch3 = y_train[inds_batch3]

        y_batch = np.abs(y_batch1 - y_batch2)
        y_batch[y_batch>0] = 1

        
        feed_dict = {X1: X_batch1, X2: X_batch2, X3: X_batch3, y: y_batch, m_test: m_test0}     
        sess.run(optimizer,feed_dict)
        loss_batch = sess.run(reduced_loss,feed_dict)    
          
        total_loss += loss_batch
    
    #print(e,total_loss)
    M_test0 = np.linspace(0,1,100)
    total_correct0 = 0
    for m_test0 in M_test0:
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
            
            feed_dict = {X1: X_batch1, X2: X_batch2, X3: X_batch2, y: y_batch, m_test: m_test0} 
            dist_ancpos = sess.run(y_preds,feed_dict) 
    
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
    #        print(preds.astype(int))
    #        print(n_false_negative)
    #        print("="*20)        
            
            total_tp += n_true_positive
            total_tn += n_true_negative        
            total_fp += n_false_positive
            total_fn += n_false_negative
    
        #print(total_correct)
        if total_correct > total_correct0:
            
            total_correct0 = copy.copy(total_correct)
            
            total_tp0 = copy.copy(total_tp)
            total_tn0 = copy.copy(total_tn)      
            total_fp0 = copy.copy(total_fp)
            total_fn0 = copy.copy(total_fn)                
            
            
    
    
    n_test_local =  n_batch_test*batch_size_test

    print('\n' + '='*40)  
    print('epoch',e,'train loss',total_loss,'val acc',total_correct0/n_test_local)  
    #print('\nepoch',e,total_loss,total_correct/n_test_local)    
  
    print('tp',total_tp0/n_test_local,'tn',total_tn0/n_test_local)
    print('fp',total_fp0/n_test_local,'fn',total_fn0/n_test_local)
        #print((total_tp+total_tn+total_fp+total_fn)/n_test_local)
    
      
    #    print('='*batch_size_test*2)
