#!/bin/python
'''
Author: Sourav Sarkar
Date: 12 September, 2021
Email: ssarkar1@ualberta.ca
Description: Implements the diffpool GNN model with added
		graph features.

'''
    
#import the standard pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F

#import pytorch geometric packages
import torch_geometric.nn as tg_nn
import torch_geometric.utils as tg_utils

#import the GNNStack module
from .pyggnnstack import *

class GNNGraphDiffPool(nn.Module):
    '''GNN model with graph convolution operation along with
    differential pooling and addition of graph features added
    to the MLP layers
    '''
    def __init__(self, node_input_dim, graph_input_dim, emb_dim=32, pool_dim=16, num_hidden_emblayer=3,
              num_hidden_poollayer=3, num_mlp_layer=[16,1], num_poollevel=3, num_clusters=[8,4,2],
              num_hidden_graphlayer=3, graph_emb_dim=16, graph_output_dim=8,
              use_batchnorm=True, use_residual=False, dropout_rate=0.3, **kwargs):
        '''Inputs:
        node_input_dim (int): dimension of the input feature vector for the nodes
        graph_input_dim (int): dimension of the input graph (event) features
        emb_dim (int): number of neurons in each hidden (and output) layer of graph embedding computation
        pool_dim (int): number of neurons in each hidden (and output) layer of graph pooling assignment
                    matrix computation
        num_hidden_emblayer (int): number of hidden embedding layer
    num_hidden_poollayer (int): numver of hidden pooling layers
    num_mlp_layer (list): list of dimensions for each MLP layers after graph convolution
    num_poollevel (int): number of levels to pool nodes into new clusters
    num_clusters (list): list of number of clusters in each level to pool to.
    num_hidden_graphlayer (int): number of hidden graph feature layers
    graph_output_dim (int): dimension of the output embedding of the graph features
    use_batchnorm (bool): whether to use batch normalization in graph convolution
    use_residual (bool): whether to use residual network in graph convolution
    dropout_rate (float): rate at which to kill random neurons for dropout regularization
    '''
        super(GNNGraphDiffPool, self).__init__(**kwargs)

        assert (num_poollevel==len(num_clusters)), f'Number of pooling layer: {num_poollayer} and cluster target {num_clusters} mismatch'

        #initialize the model settings
        self.kernel = GaussKernel()
        self.gnn_emb = nn.ModuleList()
        self.gnn_pool = nn.ModuleList()
        self.graph_mlp = nn.ModuleList()
        self.last_mlp = nn.Sequential()

        #build the GNN (with diff pool) module
        for i in range(num_poollevel):
            is_first_level = i==0
            if is_first_level:
            gnn_inputdim = node_input_dim
            else:
                gnn_inputdim = emb_dim

        hidden_embgraph_dim = [emb_dim]*num_hidden_emblayer
        hidden_poolgraph_dim = [pool_dim]*num_hidden_poollayer

        self.gnn_emb.append(GNNStack(gnn_inputdim, hidden_graph_dim = hidden_embgraph_dim,
                          output_dim = emb_dim, dropout_rate = dropout_rate,
                          use_batchnorm = use_batchnorm, use_residual = use_residual))

        self.gnn_pool.append(GNNStack(gnn_inputdim, hidden_graph_dim = hidden_poolgraph_dim,
                           output_dim = num_clusters[i], output_activation='SoftMax', dropout_rate = False,
                           use_batchnorm = use_batchnorm, use_residual = use_residual))

        #build the MLP for graph features
        in_dim = [graph_input_dim]+[graph_emb_dim]*num_hidden_graphlayer
        out_dim = [graph_emb_dim]*num_hidden_graphlayer+[graph_output_dim]
        for idx, (indim, outdim) in enumerate(zip(in_dim, out_dim)):
            is_last_layer = idx==num_hidden_graphlayer
            self.graph_mlp.append(nn.Linear(indim, outdim, bias=True))
            if not(is_last_layer) and use_batchnorm:
                self.graph_mlp.append(nn.BatchNorm1d(outdim))
            self.graph_mlp.append(nn.ReLU())
            if not(is_last_layer) and dropout_rate:
                self.graph_mlp.append(nn.Dropout(dropout_rate))


        #build last MLP layer
        in_dim = [emb_dim+graph_output_dim]+num_mlp_layer[:-1]
        out_dim = num_mlp_layer
        for idx, (indim, outdim) in enumerate(zip(in_dim, out_dim)):
            is_last_layer = idx == (len(num_mlp_layer)-1)
            if not(is_last_layer):
                self.last_mlp.add_module("embeddingtohidden", nn.Linear(indim, outdim, bias=True))
                self.last_mlp.add_module("AddReLU", nn.ReLU())
            else:
                self.last_mlp.add_module("hiddentooutput", nn.Linear(indim, outdim, bias=True))
                self.last_mlp.add_module("AddSigmoid", nn.Sigmoid())

    def forward(self, data):
        #get individual data objects
        x, edge_index, edge_pos, batch = data.x, data.edge_index, data.pos, data.batch
        edge_weight = self.kernel(edge_pos, edge_index)
        # initialize the link_loss and entropy_loss
        lsum = 0
        esum = 0
        for i in range(len(self.gnn_emb)):
            #compute pooling gcn (assignment matrix)
            s = self.gnn_pool[i](x, edge_index, edge_weight)
            x = self.gnn_emb[i](x, edge_index, edge_weight)

            #get the cluster and feature dimension for later use (dense to sparse x matrix conversion)
            _, num_clusters = s.size()
            _, num_features = x.size()

            #convert matrices from sparse to dense for input into diff pool operation
            if i==0:
                x_dense, mask = tg_utils.to_dense_batch(x, batch=batch)
                s_dense, _    = tg_utils.to_dense_batch(s, batch=batch)
                adj_dense     = tg_utils.to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)

            else:
                x_dense = x.view(num_batch, num_nodes, num_features)
                s_dense = s.view(num_batch, num_nodes, num_clusters)
                mask = None
                adj_dense = adj_pool

            #perform diffpool operation
            x_pool, adj_pool, link_loss, entr_loss = tg_nn.dense_diff_pool(x_dense, adj_dense, s_dense, mask=mask)
            #add the pool losses
            lsum += link_loss
            esum += entr_loss

            #convert the matrices back to sparse for use in next iteration
            edge_index, edge_weight = tg_utils.dense_to_sparse(adj_pool)
            num_batch, num_nodes, _ = x_pool.size()
            x = x_pool.view(num_batch*num_nodes, num_features)

        #global mean pool at the end of diffpool levels
        x = x_pool.mean(dim=1)
    

        #compute the graph features mlp
        G = data.graphx
        G, _ = tg_utils.to_dense_batch(G, batch=data.graphx_batch)
        for layer in self.graph_mlp:
            G = layer(G)

        #Concatenate the pooled node embeddings and graph embeddings
        x = torch.cat([x, G], -1)

        #Compute the last fully connected layers
        x = self.last_mlp(x)

        return x, lsum, esum

