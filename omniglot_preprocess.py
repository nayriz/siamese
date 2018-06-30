#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:15:28 2018

@author: john
"""

import imageio
import os
import numpy as np
import json

path1 = '/media/john/github/siamese/omniglot/python/images_background/'

listdir1 = os.listdir(path1)
listdir1.sort()
train_list = []
letter_dict = {} 
y_train = []
letter_class = 0
letter_number = 0

for alphabet in listdir1 :
    
    listdir_tmp = os.listdir(path1 + alphabet)    
    listdir_tmp.sort()    
    
    internal_letter_class = 0    
    for letter in listdir_tmp:
        
        path2 = path1 + '/'+ alphabet + '/' + letter + '/'
        filelist = os.listdir(path2)
        filelist.sort()
            

        for sample in filelist :
                
            y_train.append(letter_class)            
            img = imageio.imread(path2 + sample)
            train_list.append(img)            
            letter_dict[letter_number] = alphabet + '_' + str(internal_letter_class)
          
            letter_number += 1
            
        internal_letter_class +=1              
        letter_class += 1
        
X_train = np.array(train_list)  
np.save('omniglot/X_train',X_train)
y_train = np.array(y_train)
np.save('omniglot/y_train',y_train)

with open('omniglot/letter_ref_train.json', 'w') as f:
    json.dump(letter_dict, f)
    
###############################################################################

path1 = '/media/john/github/siamese/omniglot/python/images_evaluation/'

listdir1 = os.listdir(path1)
listdir1.sort()
test_list = []
letter_dict = {} 
y_test = []
letter_class = 0
letter_number = 0

for alphabet in listdir1 :
    
    listdir_tmp = os.listdir(path1 + alphabet)    
    listdir_tmp.sort()    
    
    internal_letter_class = 0    
    for letter in listdir_tmp:
        
        path2 = path1 + '/'+ alphabet + '/' + letter + '/'
        filelist = os.listdir(path2)
        filelist.sort()
            

        for sample in filelist :
                
            y_test.append(letter_class)            
            img = imageio.imread(path2 + sample)
            test_list.append(img)            
            letter_dict[letter_number] = alphabet + '_' + str(internal_letter_class)
          
            letter_number += 1
            
        internal_letter_class +=1              
        letter_class += 1
        
X_test = np.array(test_list)  
np.save('omniglot/X_test',X_test)
y_test = np.array(y_test)
np.save('omniglot/y_test',y_test)

with open('omniglot/letter_ref_test.json', 'w') as f:
    json.dump(letter_dict, f)
    