# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:41:42 2022

@author: abder
"""

import pickle
import time
import argparse
from data.SBMs import SBMsDatasetDGL 


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        
def prepare_data(args):
      
    start = time.time()

    DATASET_NAME = 'p_' + str(args.p1) + str(args.p2)
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



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Generate')
    
    parser.add_argument('--p1', type = float, required=True,
                        help='Density inside the first community (cf stochastic block model)')
    parser.add_argument('--p2',type = float,required = False,
                        help='Density inside the second community. If not specified it is equal to p1')

        
    args = parser.parse_args()
    
    if args.p2 is None :
        args.p2 = args.p1 
    
    prepare_data(args)