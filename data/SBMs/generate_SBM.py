# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:15:01 2022

@author: abder
"""
import numpy as np
import torch
import pickle
import time
import tqdm
import sys



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
        q_pattern = SBM_parameters['q_pattern']
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
        
        
# Generate and save SBM graphs
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self



start = time.time()
# configuration for 100 patterns 100/20 
nb_pattern_instances = 1 # nb of patterns
nb_train_graphs_per_pattern_instance = 100 # train per pattern
nb_test_graphs_per_pattern_instance = 20 # test, val per pattern



SBM_parameters = {}
SBM_parameters['size_min'] = 35
SBM_parameters['size_max'] = 60 
SBM_parameters['p_pattern'] = float (sys.argv[1])
SBM_parameters['q_pattern'] = 0.1 ##0.25     
SBM_parameters['vocab_size'] = 3
print(SBM_parameters)
    

dataset_train = []
dataset_val = []
dataset_test = []
for idx in tqdm.tqdm(range(nb_pattern_instances)):
    
    print('pattern:',idx)
    SBM_parameters['W0'] = random_pattern(SBM_parameters['size_min'],SBM_parameters['size_max'],SBM_parameters['p_pattern'])
    SBM_parameters['u0'] = np.random.randint(SBM_parameters['vocab_size'],size=SBM_parameters['W0'].shape[0])
    
    for _ in range(nb_train_graphs_per_pattern_instance):
        SBM_parameters['W1'] = random_pattern(SBM_parameters['size_min'],SBM_parameters['size_max'],SBM_parameters['p_pattern'])
        SBM_parameters['u1'] = np.random.randint(SBM_parameters['vocab_size'],size=SBM_parameters['W1'].shape[0])
        data = generate_SBM_graph(SBM_parameters)
        graph = DotDict()
        graph.nb_nodes = data.nb_nodes
        graph.W = data.W
        graph.rand_idx = data.rand_idx
        graph.node_feat = data.node_feat
        graph.node_label = data.node_label
        dataset_train.append(graph)

    for _ in range(nb_test_graphs_per_pattern_instance):
        SBM_parameters['W1'] = random_pattern(SBM_parameters['size_min'],SBM_parameters['size_max'],SBM_parameters['p_pattern'])
        SBM_parameters['u1'] = np.random.randint(SBM_parameters['vocab_size'],size=SBM_parameters['W1'].shape[0])
        data = generate_SBM_graph(SBM_parameters)
        graph = DotDict()
        graph.nb_nodes = data.nb_nodes
        graph.W = data.W
        graph.rand_idx = data.rand_idx
        graph.node_feat = data.node_feat
        graph.node_label = data.node_label
        dataset_val.append(graph)

    for _ in range(nb_test_graphs_per_pattern_instance):
        SBM_parameters['W1'] = random_pattern(SBM_parameters['size_min'],SBM_parameters['size_max'],SBM_parameters['p_pattern'])
        SBM_parameters['u1'] = np.random.randint(SBM_parameters['vocab_size'],size=SBM_parameters['W1'].shape[0])
        data = generate_SBM_graph(SBM_parameters)
        graph = DotDict()
        graph.nb_nodes = data.nb_nodes
        graph.W = data.W
        graph.rand_idx = data.rand_idx
        graph.node_feat = data.node_feat
        graph.node_label = data.node_label
        dataset_test.append(graph)


print(len(dataset_train),len(dataset_val),len(dataset_test))





with open('p_'+str(SBM_parameters['p_pattern'])+'_train.pkl',"wb") as f:
    pickle.dump(dataset_train,f)
with open('p_'+str(SBM_parameters['p_pattern'])+'_val.pkl',"wb") as f:
    pickle.dump(dataset_val,f)
with open('p_'+str(SBM_parameters['p_pattern'])+'_test.pkl',"wb") as f:
    pickle.dump(dataset_test,f)


    
print('Time (sec):',time.time() - start) 
