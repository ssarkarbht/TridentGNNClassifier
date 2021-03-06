#!/bin/python

#build the final pre-shuffle pytorch dataset originating from hdf5 files

#class in this script inherits from base class in hd5
from .hd5 import *

import torch_geometric.data as tg_data

class H5ToPygGraph(ShuffledTorchHD5Dataset):
	'''
	Class to represent pre-shuffled pytorch dataset in graph structure
	originating from HDF5 file
	'''
	def __init__(self, filepath, features, edgeparams, balance_dataset=False, min_track_length=None, max_cascade_energy=None, flavors=None, currents=None, memmap_directory='./memmaps', class_weights=None, close_file=False):
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
		self.feature_names = features
		self.edgeparam_names = edgeparams

		targets = self._create_targets()
		self._idxs = self._create_idxs(targets, balance_dataset, min_track_length, max_cascade_energy, flavors, currents)
		number_vertices = np.array(self.file['VertexNumber'])
		event_offsets = (number_vertices.cumsum() - number_vertices)[self._idxs]
		self.number_vertices = number_vertices[self._idxs]
		self.event_offsets = self.number_vertices.cumsum() - self.number_vertices # 'self.event_offests' refers to the feature matrix, while 'event_offsets' refers to the hd5 file
		self.targets = targets[self._idxs]

		# Create memmaps for features and edge parameters for faster access during training
		os.makedirs(os.path.dirname(memmap_directory), exist_ok=True)

		# Load precomputed memmaps based on the hash of the columns, filename and index set
		self._idxs_hash = hashlib.sha1(self._idxs.data).hexdigest()
		features_hash = hashlib.sha1(str([[os.path.relpath(filepath)] + features]).encode()).hexdigest()
		edgeparams_hash = hashlib.sha1(str([[os.path.relpath(filepath)] + edgeparams]).encode()).hexdigest()

		feature_memmap_path = os.path.join(memmap_directory, f'hd5_features_{features_hash}_{self._idxs_hash}')
		edgeparam_memmap_path = os.path.join(memmap_directory, f'hd5_edgeparams_{edgeparams_hash}_{self._idxs_hash}')

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


		#sanity checks
		endpoints = self.number_vertices + self.event_offsets
		assert(np.max(endpoints)) <= self.features.shape[0]

		if close_file:
			self.file.close()
			self.file = filepath

	def __len__(self):
		return self.number_vertices.shape[0]

	def __getitem__(self, idx):
		N = self.number_vertices[idx]
		offset = self.event_offsets[idx]
		X = self.features[offset : offset + N]
		C = self.edgeparams[offset : offset + N]
		y = self.targets[idx]
		E = np.zeros((2, N*(N-1)))
		E[0] = np.concatenate([np.repeat(i,(N-1)) for i in np.arange(N)])
		E[1] = np.concatenate([np.delete(np.arange(N),i) for i in np.arange(N)])

		if torch.cuda.is_available():
			device=torch.device('cuda')
		else:
			device=torch.device('cpu')
		tx = torch.FloatTensor(X).to(device)
		tc = torch.FloatTensor(C).to(device)
		ty = torch.FloatTensor([y]).to(device)
		te = torch.LongTensor(E).to(device)
        
		data = tg_data.Data(x=tx, edge_index=te, y=ty, pos=tc)
		return data

