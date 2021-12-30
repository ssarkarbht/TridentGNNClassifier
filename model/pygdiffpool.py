#!/bin/python

'''
Author: Sourav Sarkar
Email: ssarkar1@ualberta.ca
Date: 17 August, 2021
Description: This script contains the modules that implements
	the graph convolutions with differential pooling.
	the module here calls the general GNNStack module
	for both graph embedding and graph pooling
'''

#import the standard pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F

#import pytorch geometric packages
import torch_geometric.nn as tg_nn
import torch_geometric.utils as tg_utils

#import the GNNSTack module
from .pyggnnstack import *

class GNNDiffPool(nn.Module):
    def __init__(self, input_dim, emb_dim=32, pool_dim=16, num_hidden_emblayer=3, num_hidden_poollayer=3,
                num_mlp_layer=[1], num_poollevel=3, num_clusters=[8,4,2], use_batchnorm=True,
                use_residual=False, dropout_rate=0.3, **kwargs):
        super(GNNDiffPool, self).__init__(**kwargs)

        assert (num_poollevel==len(num_clusters)), f'Number of pooling layer: {num_poollayer} and cluster target {num_clusters} mismatch'
        #initialize the model settings
        self.kernel   = GaussKernel()
        self.gnn_emb  = nn.ModuleList()
        self.gnn_pool = nn.ModuleList()
        self.lin_mlp  = nn.Sequential()

        for i in range(num_poollevel):
            is_first_level = i==0
            if is_first_level:
                emb_input = input_dim
                pool_input= input_dim
            else:
                emb_input = emb_dim
                pool_input= pool_dim

            hidden_embgraph_dim = [emb_dim]*num_hidden_emblayer
            hidden_poolgraph_dim= [pool_dim]*num_hidden_poollayer

            self.gnn_emb.append(GNNStack(emb_input, hidden_graph_dim = hidden_embgraph_dim, 
                                output_dim = emb_dim, dropout_rate = dropout_rate,
                                use_batchnorm = use_batchnorm, use_residual = use_residual))


            self.gnn_pool.append(GNNStack(emb_input, hidden_graph_dim = hidden_poolgraph_dim, 
                                output_dim = num_clusters[i], output_activation=False, dropout_rate = False,
                                use_batchnorm = use_batchnorm, use_residual = use_residual))

        for idx, (d_in, d_out) in enumerate(zip([emb_dim]+num_mlp_layer[:-1], num_mlp_layer)):
            is_last_layer = idx == (len(num_mlp_layer) - 1)
            if not is_last_layer:
                self.lin_mlp.add_module("hiddetofullc", nn.Linear(d_in, d_out, bias=True))
                self.lin_mlp.add_module("AddRelu", nn.ReLU())
            elif is_last_layer:
                self.lin_mlp.add_module("hiddentooutput", nn.Linear(d_in, d_out, bias=True))
                self.lin_mlp.add_module("AddSigmoid", nn.Sigmoid())

    def forward(self, data):
        #get indiviual data objects
        x, edge_index, edge_pos, batch = data.x, data.edge_index, data.pos, data.batch
        #edge_weight = GaussWeight(edge_pos, edge_index)
        edge_weight = self.kernel(edge_pos, edge_index)
        # initialize the link_loss and entropy_loss
        lsum = 0
        esum = 0
        for i in range(len(self.gnn_emb)):
            if i==0:
                s = self.gnn_pool[i](x, edge_index, edge_weight)
                x = self.gnn_emb[i](x, edge_index, edge_weight)
                _, num_clusters = s.size()
                _, num_features = x.size()
                #convert the matrices from sparse to dense
                x_dense, mask = tg_utils.to_dense_batch(x, batch=batch)
                adj_dense     = tg_utils.to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)
                s_dense, _    = tg_utils.to_dense_batch(s, batch=batch)
            else:
                s = self.gnn_pool[i](x, pool_index, pool_weight)
                x = self.gnn_emb[i](x, pool_index, pool_weight)
                _, num_clusters = s.size()
                _, num_features = x.size()
                #manually convert the matrices from sparse to dense from 2nd level onwards
                x_dense = x.view(num_batch, num_nodes, num_features)
                s_dense = s.view(num_batch, num_nodes, num_clusters)
                mask = None
                adj_dense = adj_pool

            #perform diff pool
            x_pool, adj_pool, link_loss, entr_loss = tg_nn.dense_diff_pool(x_dense, adj_dense, s_dense, mask=mask)
            #add the losses
            lsum += link_loss
            esum += entr_loss
            #convert them back to sparse matrices
            pool_index, pool_weight = tg_utils.dense_to_sparse(adj_pool)
            num_batch, num_nodes, _ = x_pool.size()
            x = x_pool.view(num_batch*num_nodes, num_features)
        x = x_pool.mean(dim=1)
        x = self.lin_mlp(x)

        return x, lsum, esum


