#!/bin/python

'''
Author: Sourav Sarkar
Email: ssarkar1@ualberta.ca
Date: August 14, 2021
Description: Module in this script is the generic GNN module
	that can be used both as standard node embedding as
	well as pooling gnn for learning cluster assignment
	matrix.
'''

#import standard pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F

#import torch geometric packages
import torch_geometric.nn as tg_nn
import torch_geometric.utils as tg_utils
import torch_geometric.data as tg_data

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_graph_dim=[32,32,32], output_dim=32, output_activation=True,
                dropout_rate=0.3, use_batchnorm=True, use_residual=False, **kwargs):
        super(GNNStack, self).__init__(**kwargs)

        #initalize the model settings 
        self.dropout_rate = dropout_rate
        self.batchnorm = use_batchnorm
        self.residual  = use_residual

        #initialize the modulelist for graph model parameters containers
        self.graph_layers = nn.ModuleList()

        #Add all the hidden layers
        for i, (indim, outdim) in enumerate(zip([input_dim]+hidden_graph_dim[:-1], hidden_graph_dim)):
            is_first_layer = i==0
            self.graph_layers.append(self.conv_builder(indim, outdim,
                                    use_batchnorm  = self.batchnorm,
                                    use_activation = True,
                                    use_residual   = self.residual and (indim == outdim),
                                    use_dropout    = True))
        #Add the last hidden layer to output layer
        self.graph_layers.append(self.conv_builder(hidden_graph_dim[-1], output_dim,
                                 use_batchnorm  = False,
                                 use_activation = output_activation,
                                 use_residual   = False,
                                 use_dropout    = False))

    def conv_builder(self, input_dim, output_dim, use_batchnorm=True, use_activation=True, activation='ReLU', use_residual=True, use_dropout=True):
        mod_list=[]
        mod_list.append((tg_nn.GCNConv(input_dim, output_dim, add_self_loops=True, normalize=True, bias=True),
                         'x, edge_index, edge_weight -> x'))
        if use_batchnorm:
            mod_list.append((tg_nn.BatchNorm(output_dim), 'x -> x'))
        if use_activation:
            if activation=='ReLU':
                mod_list.append(nn.ReLU())
            elif activation=='SoftMax':
                mod_list.append(nn.Softmax(dim=1))
        if use_dropout:
            mod_list.append((nn.Dropout(self.dropout_rate), 'x -> x'))
        conv_layer = tg_nn.Sequential('x, edge_index, edge_weight', mod_list)
        return conv_layer

    def forward(self, x, edge_index, edge_weight):
        for graph in self.graph_layers:
            x = graph(x, edge_index, edge_weight)
#            emb = x
        return x
#        return emb, x

class GaussKernel(nn.Module):
    ''' This class computes the edge weights from the node coordinates
    and the sigmas of the gaussian function are the learnable parameters
    '''
    def __init__(self, *args, **kwargs):
        super(GaussKernel, self).__init__(*args, **kwargs)
        self.inverse_sigma1 = nn.Parameter(torch.rand(1) * 1e-4  + 1e-3)
        self.inverse_sigma2 = nn.Parameter(torch.rand(1)  + 1.00)

    def forward(self, edge_coordinate, edge_index):
        D, T = get_edge_weight(edge_coordinate, edge_index)
        A    = torch.exp(-(D * self.inverse_sigma1**2 + T * self.inverse_sigma2**2))
        return A

def get_edge_weight(edge_coordinate, edge_index):
    ''' compute the weight factors (numerator) of the gaussian function's 
    '''
    if torch.cuda.is_available():
        device='cuda'
    else: device = 'cpu'

    ind1 = torch.tensor([0,1,2],dtype=torch.long).to(device=device)
    ind2 = torch.tensor([3,4,5],dtype=torch.long).to(device=device)

    pos1 = edge_coordinate.index_select(dim=0, index=edge_index[0]).index_select(dim=1, index=ind1)
    pos2 = edge_coordinate.index_select(dim=0, index=edge_index[1]).index_select(dim=1, index=ind1)

    ang1 = edge_coordinate.index_select(dim=0, index=edge_index[0]).index_select(dim=1, index=ind2)
    ang2 = edge_coordinate.index_select(dim=0, index=edge_index[1]).index_select(dim=1, index=ind2)

    D = ((pos1-pos2)**2).sum(-1)
    T = (1-(ang1*ang2).sum(-1))**2
    return (D, T)

def GaussWeight(edge_coordinate, edge_index):
    inverse_sigma1 = 0.08
    inverse_sigma2 = 0.9
    D, T = get_edge_weight(edge_coordinate, edge_index)
    A    = torch.exp(-(D * inverse_sigma1**2 + T * inverse_sigma2**2))
    #print (D, T)
    #print (A)
    return A

