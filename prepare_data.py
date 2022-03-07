# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:41:42 2022

@author: abder
"""

import pickle
import time
import sys
from data.SBMs import SBMsDatasetDGL 


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
        
start = time.time()
DATASET_NAME = 'p_' + sys.argv[1]
dataset = SBMsDatasetDGL(DATASET_NAME) 
print('Time (sec):',time.time() - start) 

print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))
print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])


start = time.time()
with open(DATASET_NAME + '.pkl','wb') as f:
        pickle.dump([dataset.train,dataset.val,dataset.test],f)
        
print('Time (sec):',time.time() - start) 