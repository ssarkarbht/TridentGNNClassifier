#!/bin/python

'''
Author: Sourav Sarkar
Email: ssarkar1@ualberta.ca
Date: June 23, 2021
Description: This script contains the vanilla model for
	graph convolutional neut=ral network using PyTorch
	geometric packages, where most of the exisiting
	graph neural network methods are already implemented
'''

#import the standard pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#from torchvision import datasets, transforms

#import pytorch geometric packages
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
#from torch_geometric.datasets import Planetoid
#import torch_geometric.transforms as T


class VanillaGCNStack(nn.Module):
    def __init__(self, input_dim, hidden_graph_dim=[32,32,32,32], hidden_fullc_dim=[], output_dim=1, task='graph',
                dropout_rate=0.3, use_batchnorm=True, use_residual=False, **kwargs):
        super(VanillaGCNStack, self).__init__(**kwargs)

        #Initialize some setting parameters
        self.task=task
        self.dropout_rate = dropout_rate
        self.batch_norm = use_batchnorm
        self.residual   = use_residual

        #initalize the model parameters in Gaussian Kernel
        self.kernel = GaussKernel()

        #initialize the modulelist for graph model parameter containers
        self.graph_layers = nn.ModuleList()
        #Add the parameter placeholders to module list for each graph layers
        #Add the first layer
        self.graph_layers.append(self.conv_builder(input_dim, hidden_graph_dim[0], use_batchnorm=self.batch_norm,
                                use_activation=True, use_residual=False, use_dropout=True))

        #Add the hiddent graph layers
        for i in range(len(hidden_graph_dim[:-1])):
            is_last_layer = i == len(hidden_graph_dim)-2
            self.graph_layers.append(self.conv_builder(hidden_graph_dim[i], hidden_graph_dim[i+1],
                                    use_batchnorm  = self.batch_norm and not is_last_layer,
                                    use_activation = not is_last_layer,
                                    use_residual   = False,
                                    use_dropout    = not is_last_layer))

        #Add post message passing layers into one sequential module
        self.postmp_layers = nn.Sequential()

        #If the task is node classification, keep the graph structure intact
        if self.task=='node':
            #Add one hidden to hidden layer without message passing
            self.postmp_layers.add_module("hiddenTohidden", nn.Linear(hidden_graph_dim[-1],hidden_graph_dim[-1]))
            self.postmp_layers.add_module("addReLU", nn.ReLU())
            #Add final layer outputing the classification score
            self.postmp_layers.add_module("hiddenToOutput", nn.Linear(hidden_graph_dim[-1], output_dim))
            self.postmp_layers.add_module("addsigmoid", nn.Sigmoid())

        #If the task is graph classification, add the layers after pooling is performed
        if self.task=='graph':
            #If non-zero hidden MLP layer, add them here
            if len(hidden_fullc_dim)!=0:
                for i in range(len(hidden_fullc_dim)):
                    #add the first pooled layer: graph embeddings to hidden MLP layer
                    if i==0:
                        self.postmp_layers.add_module("graphTofullyc", nn.Linear(hidden_graph_dim[-1], hidden_fullc_dim[i], bias=True))
                        self.postmp_layers.add_module("addReLU", nn.relu())
                    #add hidden MLP to next hidden MLP
                    elif 0<i<(len(hidden_fullc_dim)-1):
                        self.postmp_layers.add_module("fullcTofullc", nn.Linear(hidden_fullc_dim[i], hidden_fullc_dim[i+1], bias=True))
                        self.postmp_layers.add_module("addReLU", nn.relu())
                    #Add the last hidden to output layer
                    else:
                        self.postmp_layers.add_module("hiddenTooutput", nn.Linear(hidden_fullc_dim[i], output_dim, bias=True))
                        self.postmp_layers.add_module("addSigmoid", nn.Sigmoid())
            #If no hidden MLP layer, add the hidden last graph layer (pooled) to output layer
            else:
                self.postmp_layers.add_module("graphTofullyc", nn.Linear(hidden_graph_dim[-1], output_dim))
                self.postmp_layers.add_module("addSigmoid", nn.Sigmoid())

        if not (self.task == 'node' or self.task== 'graph'):
            raise RuntimeError('Task not specified.')


    def conv_builder(self, input_dim, output_dim, use_batchnorm=True, use_activation=True, use_residual=False, use_dropout=False):
        mod_list = []
        mod_list.append((pyg_nn.GCNConv(input_dim, output_dim), 'x, edge_index, edge_weight -> x'))
        if use_batchnorm:
            mod_list.append((pyg_nn.BatchNorm(output_dim), 'x -> x'))
        if use_activation:
            mod_list.append(nn.ReLU())
#        if use_residual
        if use_dropout:
            mod_list.append((nn.Dropout(self.dropout_rate), 'x -> x'))
        conv_layer = pyg_nn.Sequential('x, edge_index, edge_weight', mod_list)
        return conv_layer


    def forward(self, data):
        x, edge_index, edge_pos, batch = data.x, data.edge_index, data.pos, data.batch
        edge_weights = self.kernel(edge_pos, edge_index)
        for i in range(len(self.graph_layers)):
            x = self.graph_layers[i](x, edge_index, edge_weights)
            emb = x
#            x = F.relu(x)
#            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.postmp_layers(x)

        return emb, x
#        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.binary_cross_entropy(pred, label)


def edge_weight_factor(edge_coordinate, edge_index):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    ind1 = torch.tensor([0,1,2],dtype=torch.long).to(device=device)
    ind2 = torch.tensor([3,4,5],dtype=torch.long).to(device=device)

    pos1 = edge_coordinate.index_select(dim=0, index=edge_index[0]).index_select(dim=1, index=ind1)
    pos2 = edge_coordinate.index_select(dim=0, index=edge_index[1]).index_select(dim=1, index=ind1)

    ang1 = edge_coordinate.index_select(dim=0, index=edge_index[0]).index_select(dim=1, index=ind2)
    ang2 = edge_coordinate.index_select(dim=0, index=edge_index[1]).index_select(dim=1, index=ind2)

    D = ((pos1-pos2)**2).sum(-1)
    T = (1-(ang1*ang2).sum(-1))**2
    return (D,T)

class GaussKernel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_sigma1 = nn.Parameter(torch.rand(1) * 0.02 + 0.99)
        self.inverse_sigma2 = nn.Parameter(torch.rand(1) * 0.02 + 0.99)

    def forward(self, edge_coordinate, edge_index):
        D, T = edge_weight_factor(edge_coordinate, edge_index)
        A    = torch.exp(-(D * self.inverse_sigma1**2 + T * self.inverse_sigma2**2))
        return A


