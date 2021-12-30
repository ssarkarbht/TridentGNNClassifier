#!/bin/python

'''
Author: Sourav Sarkar
Date: June 15, 2021
Email: ssarkar1@ualberta.ca
Objective: This script uses the classes and functions from utildata
	and model to apply the settings parameters and prepare the neural
	network.
'''

import json
from utildata import *
from model import *
from collections import Mapping

def dict_update(d, u):
	'''
	Recursively updates a dictionary with another. Used for 
	parsing settings for training.

	Parameters:
	d [dict]: The dict to update
	u [dict]: The dict that contains keys that should be updated in d
	'''

	for key in u:
		if key in d:
			if isinstance(d[key], Mapping):
				dict_update(d[key], u[key])
			else:
				d[key] = u[key]
		else:
			raise RuntimeError(f'Unknown setting {key}')

def dataset_from_config(config, filter_non_train=False, close_file=False):
	'''
	Creates a dataset from a configuration file

	Parameters:
	config [dict]: The configuration dict
	filter_non_train [bool]: If true, validation and testing dataset are filtered
	close_file [bool]: If True, hdf5 files will be closed after readout.

	Returns:
	train : utildata.ShuffledTorchHD5Dataset
	val   : utildata.ShuffledTorchHD5Dataset
	test  : utildata.ShuffledTorchHD5Dataset
	'''
	dataset_config = config['dataset']
	dataset_type   = dataset_config['type'].lower()
	if dataset_type in ('h5', 'hd5', 'hdf5'):
		train = ShuffledGraphTorchHD5Dataset(
                      dataset_config['paths']['train'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      balance_dataset = dataset_config['balance_classes'],
                      min_track_length = dataset_config['min_track_length'],
                      max_cascade_energy = dataset_config['max_cascade_energy'],
                      flavors = dataset_config['flavors'],
                      currents = dataset_config['currents'],
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)
                      
		val = ShuffledGraphTorchHD5Dataset(
                      dataset_config['paths']['validation'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      balance_dataset = False,
                      min_track_length = dataset_config['min_track_length'] if filter_non_train else None,
                      max_cascade_energy = dataset_config['max_cascade_energy'] if filter_non_train else None,
                      flavors = dataset_config['flavors'] if filter_non_train else None,
                      currents = dataset_config['currents'] if filter_non_train else None,
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)

		test = ShuffledGraphTorchHD5Dataset(
                      dataset_config['paths']['test'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      balance_dataset = False,
                      min_track_length = dataset_config['min_track_length'] if filter_non_train else None,
                      max_cascade_energy = dataset_config['max_cascade_energy'] if filter_non_train else None,
                      flavors = dataset_config['flavors'] if filter_non_train else None,
                      currents = dataset_config['currents'] if filter_non_train else None,
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)
		return train, val, test
	else:
		raise RuntimeError(f'Unknown dataset type {dataset_type}')


def pygdataset_from_config(config, filter_non_train=False, close_file=False):
	'''
	Creates a dataset from a configuration file

	Parameters:
	config [dict]: The configuration dict
	filter_non_train [bool]: If true, validation and testing dataset are filtered
	close_file [bool]: If True, hdf5 files will be closed after readout.

	Returns:
	train : utildata.ShuffledTorchHD5Dataset
	val   : utildata.ShuffledTorchHD5Dataset
	test  : utildata.ShuffledTorchHD5Dataset
	'''
	dataset_config = config['dataset']
	dataset_type   = dataset_config['type'].lower()
	if dataset_type in ('h5', 'hd5', 'hdf5'):
		train = H5ToPygGraph(
                      dataset_config['paths']['train'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      balance_dataset = dataset_config['balance_classes'],
                      min_track_length = dataset_config['min_track_length'],
                      max_cascade_energy = dataset_config['max_cascade_energy'],
                      flavors = dataset_config['flavors'],
                      currents = dataset_config['currents'],
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)
                      
		val = H5ToPygGraph(
                      dataset_config['paths']['validation'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      balance_dataset = False,
                      min_track_length = dataset_config['min_track_length'] if filter_non_train else None,
                      max_cascade_energy = dataset_config['max_cascade_energy'] if filter_non_train else None,
                      flavors = dataset_config['flavors'] if filter_non_train else None,
                      currents = dataset_config['currents'] if filter_non_train else None,
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)

		test = H5ToPygGraph(
                      dataset_config['paths']['test'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      balance_dataset = False,
                      min_track_length = dataset_config['min_track_length'] if filter_non_train else None,
                      max_cascade_energy = dataset_config['max_cascade_energy'] if filter_non_train else None,
                      flavors = dataset_config['flavors'] if filter_non_train else None,
                      currents = dataset_config['currents'] if filter_non_train else None,
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)
		return train, val, test
	elif dataset_type in ('h5_withgraph', 'hd5_withgraph', 'hdf5_withgraph'):
		train = H5ToPygGraphWithGraphFeatures(
                      dataset_config['paths']['train'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      graph_features = dataset_config['graph_features'],
                      balance_dataset = dataset_config['balance_classes'],
                      min_track_length = dataset_config['min_track_length'],
                      max_cascade_energy = dataset_config['max_cascade_energy'],
                      flavors = dataset_config['flavors'],
                      currents = dataset_config['currents'],
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)
                      
		val = H5ToPygGraphWithGraphFeatures(
                      dataset_config['paths']['validation'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      graph_features = dataset_config['graph_features'],
                      balance_dataset = False,
                      min_track_length = dataset_config['min_track_length'] if filter_non_train else None,
                      max_cascade_energy = dataset_config['max_cascade_energy'] if filter_non_train else None,
                      flavors = dataset_config['flavors'] if filter_non_train else None,
                      currents = dataset_config['currents'] if filter_non_train else None,
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)

		test = H5ToPygGraphWithGraphFeatures(
                      dataset_config['paths']['test'],
                      features = dataset_config['features'],
                      edgeparams = dataset_config['edgeparams'],
                      graph_features = dataset_config['graph_features'],
                      balance_dataset = False,
                      min_track_length = dataset_config['min_track_length'] if filter_non_train else None,
                      max_cascade_energy = dataset_config['max_cascade_energy'] if filter_non_train else None,
                      flavors = dataset_config['flavors'] if filter_non_train else None,
                      currents = dataset_config['currents'] if filter_non_train else None,
                      class_weights = dataset_config['class_weights'],
                      close_file = close_file,)
		return train, val, test
	else:
		raise RuntimeError(f'Unknown dataset type {dataset_type}')

def model_from_config(config):
	'''
	Creates a model based on the model config file
	Parameters:
	config [dict]: The configuration file containing model parameters

	Returnd:
	model [torch.nn.model]: A PyTorch model
	'''
	number_input_features = len(config['dataset']['features'])
	model_config = config['model']
	model_type = model_config['type'].lower()
	if model_type in ('gcn', 'gcnn'):
		model = GraphConvolutionalNetwork(
                      number_input_features,
                      units_graph_convolutions = model_config['hidden_units_graph_convolutions'],
                      units_fully_connected = model_config['hidden_units_fully_connected'],
                      use_batchnorm = model_config['use_batchnorm'],
                      dropout_rate = model_config['dropout_rate'],
                      use_residual = model_config['use_residual'],)
	elif model_type in ('pyg_gcn'):
		model = VanillaGCNStack(
                    number_input_features,
                    hidden_graph_dim = model_config['hidden_units_graph_convolutions'],
                    hidden_fullc_dim = model_config['hidden_units_fully_connected'],
                    output_dim       = model_config['output_dimension'],
                    task             = model_config['classification_task'],
                    dropout_rate     = model_config['dropout_rate'],
                    use_batchnorm    = model_config['use_batchnorm'],
                    use_residual     = model_config['use_residual'])
	elif model_type == 'pyg_diffpool':
		model = GNNDiffPool(
                    number_input_features,
                    emb_dim             = model_config['embed_dim'],
                    pool_dim            = model_config['pool_dim'],
                    num_hidden_emblayer = model_config['hidden_embed_layers'],
                    num_hidden_poollayer= model_config['hidden_pool_layers'],
                    num_mlp_layer       = model_config['units_fully_connected'],
                    num_poollevel       = model_config['num_pool_levels'],
                    num_clusters        = model_config['units_clusters'],
                    use_batchnorm       = model_config['use_batchnorm'],
                    use_residual        = model_config['use_residual'],
                    dropout_rate        = model_config['dropout_rate'])
	elif model_type == 'pyg_diffpool_graphfeature':
		number_graph_features = len(config['dataset']['graph_features'])
		model = GNNGraphDiffPool(
                    number_input_features,
                    number_graph_features,
                    emb_dim             = model_config['embed_dim'],
                    pool_dim            = model_config['pool_dim'],
                    num_hidden_emblayer = model_config['hidden_embed_layers'],
                    num_hidden_poollayer= model_config['hidden_pool_layers'],
                    num_mlp_layer       = model_config['units_fully_connected'],
                    num_poollevel       = model_config['num_pool_levels'],
                    num_clusters        = model_config['units_clusters'],
                    num_hidden_graphlayer = model_config['hidden_graphlayer'],
                    graph_emb_dim       = model_config['graph_emb_dim'],
                    graph_output_dim    = model_config['graph_output_dim'],
                    use_batchnorm       = model_config['use_batchnorm'],
                    use_residual        = model_config['use_residual'],
                    dropout_rate        = model_config['dropout_rate'])
	else:
		raise RuntimeError(f'Unknown model type {model_type}')
	return model
