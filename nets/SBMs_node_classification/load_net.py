"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.graph_transformer_net import GraphTransformerNet
from nets.SBMs_node_classification.graph_transformer_net_concat import GraphTransformerNetConcat

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def GraphTransformerConcat(net_params):
    return GraphTransformerNetConcat(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer,
        'GraphTransformerConcat':GraphTransformerConcat
    }
        
    return models[MODEL_NAME](net_params)