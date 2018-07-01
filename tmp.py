#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 23:39:08 2018

@author: john
"""

###############################################################################
inds_test1 = inds_test.copy()
inds_test2 = inds_test.copy()

np.random.shuffle(inds_test1)
np.random.shuffle(inds_test2)  

y_test1 = y_test[inds_test1]
y_test2 = y_test[inds_test2]


y_comp = np.abs(y_test1 - y_test2)/np.max(y_test)

e1 = np.sum(np.abs(y_comp - np.zeros_like(y_comp)))/len(y_test)
e2 = np.sum(np.abs(y_comp - np.ones_like(y_comp)))/len(y_test)


print('The error is at most',np.min([e1,e2]))


#########################################################################