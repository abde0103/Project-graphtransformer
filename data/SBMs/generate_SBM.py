# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:15:01 2022

@author: abder
"""
import numpy as np
import pickle
import argparse
import time
import tqdm



from utils_generate_SBM import random_pattern, generate_SBM_graph

        
        
# Generate and save SBM graphs
class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def build_SBM_params(args): 
    SBM_parameters = {}
    
    SBM_parameters['p1'] = args.p1
    
    if args.q is not None :
        SBM_parameters['q'] = args.q
    
    if args.size_min is not None:    
        SBM_parameters['size_min'] = args.size_min
    
    if args.size_max is not None :    
        SBM_parameters['size_max'] = args.size_max
    
    if args.p2 is not None :
        SBM_parameters['p2'] = args.p2
    else :
        SBM_parameters['p2'] = SBM_parameters['p1']
        
    if args.size_min_test is not None :
        SBM_parameters['size_min_test'] = args.size_min_test
    else :
        SBM_parameters['size_min_test'] = SBM_parameters['size_min']
        
    if args.size_max_test is not None :
        SBM_parameters['size_max_test'] = args.size_max_test
    else :
        SBM_parameters['size_max_test'] = SBM_parameters['size_max']
    
    if args.vocab_size is not None :
        SBM_parameters['vocab_size'] = args.vocab_size
        
    print(SBM_parameters)
    
    return (SBM_parameters)


def generate(SBM_parameters):
    
    start = time.time()
    
     # nb of patterns
    if args.number_instances is not None:    
        nb_pattern_instances = args.number_instances
    nb_train_graphs_per_pattern_instance = 100 # train per pattern
    nb_test_graphs_per_pattern_instance = 20 # test, val per pattern

    dataset_train = []
    dataset_val = []
    dataset_test = []
    for idx in tqdm.tqdm(range(nb_pattern_instances)):
    
        print('pattern:',idx)
        SBM_parameters['W0'] = random_pattern(SBM_parameters['size_min'],SBM_parameters['size_max'],SBM_parameters['p1'])
        SBM_parameters['u0'] = np.random.randint(SBM_parameters['vocab_size'],size=SBM_parameters['W0'].shape[0])
        
        for _ in range(nb_train_graphs_per_pattern_instance):
            SBM_parameters['W1'] = random_pattern(SBM_parameters['size_min'],SBM_parameters['size_max'],SBM_parameters['p2'])
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
            SBM_parameters['W1'] = random_pattern(SBM_parameters['size_min'],SBM_parameters['size_max'],SBM_parameters['p2'])
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
            SBM_parameters['W1'] = random_pattern(SBM_parameters['size_min_test'],SBM_parameters['size_max_test'],SBM_parameters['p2'])
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





    with open('p_'+str(SBM_parameters['p1'])+str(SBM_parameters['p2'])+'_train.pkl',"wb") as f:
        pickle.dump(dataset_train,f)
    with open('p_'+str(SBM_parameters['p1'])+str(SBM_parameters['p2'])+'_val.pkl',"wb") as f:
        pickle.dump(dataset_val,f)
    with open('p_'+str(SBM_parameters['p1'])+str(SBM_parameters['p2'])+'_test.pkl',"wb") as f:
        pickle.dump(dataset_test,f)


    
    print('Time (sec):',time.time() - start) 






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate')
    
    parser.add_argument('--p1', type = float, required=True,
                        help='Density inside the first community (cf stochastic block model)')
    parser.add_argument('--q', type = float, required=False, default = 0.1,
                        help='Density between nodes of each community and the rest of nodes (cf stochastic block model)')
    parser.add_argument('--p2',type = float,required = False,
                        help='Density inside the second community. If not specified it is equal to p1')
    parser.add_argument('--size_min',type = int,required = False, default = 50,
                        help='Minimal number of nodes in each community')
    parser.add_argument('--size_max',type = int,required = False, default = 50,
                        help='Maximal number of nodes in each community. Fix maximal_size = minimal_size if you want deterministic number of nodes')
    parser.add_argument('--number_instances',type = int,required = False,default = 10,
                        help = 'Number of pattern instances : nb_train_graphs = 100*number_instances, nb_test_graphs = 20*number_instances')
    parser.add_argument('--vocab_size',type = int,required = False,default = 3,
                        help = 'Vocab size for nodes feature embedding just before the attention layers')


    parser.add_argument('--size_min_test',type = int,required = False,
                        help='Minimal number of nodes in each community for test graphs. If not precised, min_size_test = min_size')
    parser.add_argument('--size_max_test',type = int,required = False,
                        help='Maximal number of nodes in each community for test graphs. If not precised, max_size_test = max_size')
    
    args = parser.parse_args()
    
    generate(build_SBM_params(args))


