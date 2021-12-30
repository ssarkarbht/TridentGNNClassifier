#!/bin/python
'''
Author: Sourav Sarkar
Date: June 21, 2021
Email: ssarkar1@ualberta.ca
Objective: This script takes the large dataset (hdf5 file)
	that has all the double- and single-track events merged,
	and shuffles the events and then splits the events into
	training, validation and test datasets, based on the 
	user controlled split fractions.
'''

import numpy as np
import h5py as h5
import os
from collections import defaultdict

#VALIDATION_FRACTION = 0.5
#TESTING_FRACTION    = 0.5

VALIDATION_FRACTION = 0.0
TESTING_FRACTION    = 0.0#0.15

def extract_dataset(f, idxs, dir, prefix, column_types):
	'''
	This function takes the shuffled selected indices along with
	hdf5 parent file object and other credntials, to create the
	subset hdf5 file that can be training, validation or testing
	'''
	#Get the vertex offset values for extracting based on idx
	vertex_offsets = f['VertexNumber']['value'].cumsum() - f['VertexNumber']['value']
	#Get the square of vertex offset to extract the edge between every vertex with each other
#	vertex_squared = f['VertexNumber']['value']**2
#	edge_offsets   = vertex_squared.cumsum() - vertex_squared

	#Number of events to extract
	N_events = idxs.shape[0]
	#Total number of vertices from all events to extract
	N_vertices = f['VertexNumber']['value'][idxs].sum()
	#Total number of edge relations to extract
#	N_edges    = vertex_squared[idxs].sum()

	#make a blank list dictionary to store the types of columns into major sections
	columns = defaultdict(list)

	with h5.File(os.path.join(dir, f'{prefix}.h5'), 'w') as outfile:
		for key in f.keys():
			if column_types[key] == 'event':
				shape = (N_events,)
				datakey = 'value'
			elif column_types[key] == 'vertex':
				shape = (N_vertices,)
				datakey = 'item'
			elif column_types[key] == 'true_tracks':
				shape = (N_events*2,)
				datakey = 'item'
			elif column_types[key] == 'coordinate':
				shape = (N_events*3,)
				datakey = 'item'
			else:
				raise RuntimeError(f'Unknown column type for key {key}: {column_types[key]}')
			columns[column_types[key]].append(key)
			#print (f'Setting dataset for key {key} ...')
			if key == 'filename':
				outfile.create_dataset(key, shape=shape, dtype=f[key].dtype)
				continue
			outfile.create_dataset(key, shape=shape, dtype=f[key].dtype[datakey])

		#create index set for every vertex for all the slected events
		vertex_idxs = np.zeros((N_vertices), dtype=np.int64)
		offset = 0
		for i, idx in enumerate(idxs):
			print (f'{i}/{len(idxs)}\r', end='\r')
			size = f['VertexNumber'][idx]['value']
			vertex_offset = vertex_offsets[idx]
			vertex_idxs[offset : offset+size] = np.arange(vertex_offset, vertex_offset+size)
			offset += size

		#Copy the 'Vertex' type data
		print(f'Copying data for column type \'vertex\'')
		for key in columns['vertex']:	
			print (f'\rCopying {key}...', end='\r')
			data = np.array(f[key]['item'])[vertex_idxs]
			outfile[key][:] = data

		#Copy the 'Event' type data
		print(f'Copying data for column type \'event\'')
		for key in columns['event']:
			print (f'\rCopying {key}...', end='\r')
			if key == 'filename':
				data = np.array(f[key])[idxs]
				outfile[key][:] = data
				continue
			data = np.array(f[key]['value'])[idxs]
			outfile[key][:] = data

		#Copy the 'TrueTracks' type data
		print(f'Copying data for column type \'true_tracks\'')
		for key in columns['true_tracks']:
			print (f'\rCopying key {key} ...', end='\r')
			data = np.array(f[key]['item']).reshape((-1,2))[idxs]
			outfile[key][:] = data.reshape((-1))

		#Copy the 'Coordinate' type data
		print(f'Copying data for column type \'coordinate\'')
		for key in columns['coordinate']:
			print (f'\rCopying key {key} ...', end='\r')
			data = np.array(f[key]['item']).reshape((-1,3))[idxs]
			outfile[key][:] = data.reshape((-1))


if __name__ == '__main__':
#	input = '../dataset/resampled_dataset_3/testval_dataset.h5'
#	output = '../dataset/resampled_dataset_3/dataset_split/testval_shuffle'
	input = '../dataset/weight_dataset_10/full_dataset.h5'
	output = '../dataset/weight_dataset_10/dataset_split/'
	os.makedirs(output, exist_ok=True)
	with h5.File(input,'r') as f:
		N_events = f['VertexNumber'].shape[0]
		N_vertex = f['VertexNumber']['value'].sum()

		#categorize each data column based on the column types
		column_types = dict()
		for key in f.keys():
			N_col = f[key].shape[0]
			if N_col == N_events:
				column_types[key] = 'event'
			elif N_col == N_vertex:
				column_types[key] = 'vertex'
			elif N_col == N_events*2:
				column_types[key] = 'true_tracks'
			elif N_col == N_events*3:
				column_types[key] = 'coordinate'
			else:
				raise RuntimeError(f'Unknown column type for key {key} with shape {N_col}')

		#shuffle the indices
		idx = np.arange(N_events)
		np.random.shuffle(idx)
		N_idx_test = int(TESTING_FRACTION * N_events)
		N_idx_test_val = int((VALIDATION_FRACTION + TESTING_FRACTION) * N_events)

		test_idx = idx[ : N_idx_test]
		val_idx  = idx[N_idx_test : N_idx_test_val]
		train_idx= idx[N_idx_test_val : ]

		print (f'#Train: {train_idx.shape[0]} -- #Validation: {val_idx.shape[0]} -- #Testing: {test_idx.shape[0]}')

		extract_dataset(f, train_idx, output, 'train', column_types)
		extract_dataset(f, test_idx,  output, 'test' , column_types)
		extract_dataset(f, val_idx,   output, 'val',   column_types)
