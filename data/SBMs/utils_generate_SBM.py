# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:30:03 2022

@author: abder
"""

import numpy as np
import torch 

def schuffle(W,c):
    # relabel the vertices at random
    idx=np.random.permutation(W.shape[0])
    W_new=W[idx,:]
    W_new=W_new[:,idx]
    c_new=c[idx]
    return W_new , c_new , idx 


def random_pattern(size_min,size_max,p):
    n = np.random.randint(size_min,size_max+1)
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if np.random.binomial(1,p)==1:
                W[i,j]=1
                W[j,i]=1     
    return W 


def add_pattern(W0,W,c,num_clus,q):
    n=W.shape[0]
    n0=W0.shape[0]
    V=(np.random.rand(n0,n) < q).astype(float)
    W_up=np.concatenate(  ( W , V.T ) , axis=1 )
    W_low=np.concatenate( ( V , W0  ) , axis=1 )
    W_new=np.concatenate( (W_up,W_low)  , axis=0)
    c0=np.full(n0,num_clus)
    c_new=np.concatenate( (c, c0),axis=0)
    return W_new,c_new


class generate_SBM_graph():

    def __init__(self, SBM_parameters): 

        # parameters
        q_pattern = SBM_parameters['q']
        W0 = SBM_parameters['W0']
        u0 = SBM_parameters['u0']
        W1 = SBM_parameters['W1']
        u1 = SBM_parameters['u1']
        
        c = np.ones(W1.shape[0], dtype= int)

        # add the subgraph to be detected
        W, c = add_pattern(W0,W1,c,q= q_pattern,num_clus = 0)
        u = np.concatenate((u1,u0),axis=0)
        
        # shuffle
        W, c, idx = schuffle(W,c)
        u = u[idx]
    
        # target
        target = c.astype(float)
    
        
        # convert to pytorch
        W = torch.from_numpy(W)
        W = W.to(torch.int8)
        idx = torch.from_numpy(idx) 
        idx = idx.to(torch.int16)
        u = torch.from_numpy(u) 
        u = u.to(torch.int16)                      
        target = torch.from_numpy(target)
        target = target.to(torch.int16)
        
        # attributes
        self.nb_nodes = W.size(0)
        self.W = W
        self.rand_idx = idx
        self.node_feat = u
        self.node_label = target