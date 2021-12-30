#!/bin/python

'''
Author: Sourav Sarkar
Date: September 25, 2021
Email: ssarkar1@ualberta.ca
Description: build the final pre-shuffled pytorch dataset originating from hdf5 files
		class in this script inherits from base class in hd5
		(base framework of the hd5 script is followed from Dominic's implementation)
N.B: this script implements the graph features along with node features in the data structure
'''

from .hd5 import *

import torch_geometric.data as tg_data

class H5ToPygGraphWithGraphFeatures(ShuffledTorchHD5Dataset):
	'''
	Class to represent pre-shuffled pytorch dataset in graph structure
	originating from HDF5 file containing node and graph features
	'''
	def __init__(self, filepath, features, edgeparams, graph_features, balance_dataset=False, min_track_length=None, max_cascade_energy=None, flavors=None, currents=None, memmap_directory='./memmaps', class_weights=None, close_file=False):
		"""
		Initializes the dataset

		Parameters:
		filepath [str]: Path/To/HDF5 file
		features [list]: A list of dataset columns in the HD5 File that represent the vertex features.
		edgeparams [list]: A list of edge parameters columns from HDF5 file that represents the edge coordinates of each vertex.
		balance_dataset [bool]: If the dataset should be balanced such that each class contains the same number of samples.
		min_track_length [float or None]: Minimal track length all track-events must have.
		max_cascade_energy [float or none]: The maximal cascade energy all track events are allowed to have.
		flavors [list or None]: Only certain neutrino flavor events will be considered if given.
		currents [list or None]: Only certain current events will be considered if given.
		memmap_directory [str]: Directory for meomory maps
		class_weights [dict or None]: Weights for each class
		close_file [bool]: If True, the HDF5 file will be closed afterwards.
		"""
		super().__init__(filepath)
		# get the keys for the feature and edge parameters
		self.feature_names = features
		self.edgeparam_names = edgeparams
		self.graphfeature_names = graph_features
		#create the class labels for each event
		targets = self._create_targets()
		#applies filters and balancing to create event indices for training
		self._idxs = self._create_idxs(targets, balance_dataset, min_track_length, max_cascade_energy, flavors, currents)
		#get the number of vertices for each event
		number_vertices = np.array(self.file['VertexNumber'])
		#get the start index of each event's vertex list 
		event_offsets = (number_vertices.cumsum() - number_vertices)[self._idxs]
		#get the vertex numbers for the events in the training set (after filtering and balancing, subset of the whole file)
		self.number_vertices = number_vertices[self._idxs]
		#get the start indices of the event vertices in the training set (subset of the entire file)
		self.event_offsets = self.number_vertices.cumsum() - self.number_vertices # 'self.event_offests' refers to the feature matrix, while 'event_offsets' refers to the hd5 file
		self.targets = targets[self._idxs]

		# Create memmaps for features and edge parameters for faster access during training
		os.makedirs(os.path.dirname(memmap_directory), exist_ok=True)

		# Load precomputed memmaps based on the hash of the columns, filename and index set
		self._idxs_hash = hashlib.sha1(self._idxs.data).hexdigest()
		features_hash = hashlib.sha1(str([[os.path.relpath(filepath)] + features]).encode()).hexdigest()
		edgeparams_hash = hashlib.sha1(str([[os.path.relpath(filepath)] + edgeparams]).encode()).hexdigest()
		graphfeatures_hash = hashlib.sha1(str([[os.path.relpath(filepath)] + graph_features]).encode()).hexdigest()

		feature_memmap_path = os.path.join(memmap_directory, f'hd5_features_{features_hash}_{self._idxs_hash}')
		edgeparam_memmap_path = os.path.join(memmap_directory, f'hd5_edgeparams_{edgeparams_hash}_{self._idxs_hash}')
		graphfeature_memmap_path = os.path.join(memmap_directory, f'hd5_graphfeatures_{graphfeatures_hash}_{self._idxs_hash}')

		if not os.path.exists(feature_memmap_path) or not os.path.exists(edgeparam_memmap_path):
			# Create an index set that operates on vertex features, which is used to build memmaps efficiently
			_vertex_idxs = np.concatenate([np.arange(start, end) for start, end in zip(event_offsets, event_offsets + self.number_vertices)]).tolist()
			number_samples = len(_vertex_idxs)

		else:
			_vertex_idxs = None
			number_samples = int(self.number_vertices.sum())

		self.features = self._create_feature_memmap(feature_memmap_path, _vertex_idxs, self.feature_names, number_samples=number_samples)
		self.edgeparams = self._create_feature_memmap(edgeparam_memmap_path, _vertex_idxs, self.edgeparam_names, number_samples=number_samples)
		self.weights = self._compute_weights(class_weights, targets)
		self.graphfeatures = self._create_feature_memmap(graphfeature_memmap_path, self._idxs.tolist(), self.graphfeature_names, number_samples=self._idxs.shape[0])


		#sanity checks
		endpoints = self.number_vertices + self.event_offsets
		assert(np.max(endpoints)) <= self.features.shape[0]

		if close_file:
			self.file.close()
			self.file = filepath

	def __len__(self):
		return self.number_vertices.shape[0]

	def __getitem__(self, idx):
		'''This function iteratively creates the graph data container for each event
		'''
		#Get the number of vertices in the event graph
		N = self.number_vertices[idx]
		#Get the starting index for the node features database
		offset = self.event_offsets[idx]
		#Get the feature vector
		X = self.features[offset : offset + N]
		#Get the node parameters that are used in the edge weight calculation
		C = self.edgeparams[offset : offset + N]
		#Get the graph features
		G = self.graphfeatures[idx]
		#get the class labels
		y = self.targets[idx]
		#create a blank adjacency matrix (COO format) for all the edges
		#Note: #edges = N^2 (i.e. all nodes are connected to each other with undirected edges)
		E = np.zeros((2, N*(N-1)))
		#Create the "from node" indices
		E[0] = np.concatenate([np.repeat(i,(N-1)) for i in np.arange(N)])
		#Create the "to node" indices
		E[1] = np.concatenate([np.delete(np.arange(N),i) for i in np.arange(N)])

		#set the device before performing some tensor operations
		if torch.cuda.is_available():
			device=torch.device('cuda')
		else:
			device=torch.device('cpu')
		#convert the numpy arrays to torch tensors
		tx = torch.FloatTensor(X).to(device)
		tc = torch.FloatTensor(C).to(device)
		tG = torch.FloatTensor(G).to(device)
		ty = torch.FloatTensor([y]).to(device)
		te = torch.LongTensor(E).to(device)
        	#put the torch tensor in torch geometric data container
		data = tg_data.Data(x=tx, edge_index=te, y=ty, pos=tc, graphx=tG)
		return data

