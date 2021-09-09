#!/bin/python
'''
Author: Sourav Sarkar
Date: June 2, 2021
Email: ssarkar1@ualberta.ca
Description: This script takes the input of multiple hdf5 files
	creates from the create features scripts and joins
	them into a single large dataset to be shuffled and
	split into training, validation and test dataset by
	shuffle_split_datast.py script.
'''

import h5py as h5
from glob import glob
import sys

# Let's use the "RunID" column to retrieve the number of events per datafile
# Any keys that have only one attributes per event can be used here

PER_EVENT_COLUMN = 'RunID'

if __name__ == '__main__':
	infile_pattern = sys.argv[1]
	outfile_name   = sys.argv[2]
	infiles = sorted(glob(infile_pattern))
	print (f'Found {len(infiles)} files that match the input file pattern.')

	#dictionary to store the total size/dimension of each key objects in hdf5 files
	dimensions = {}
	#dictionary to store the dtype of each key objects in hdf5 files
	dtypes     = {}

	#Calculate the dataset total size beforehand
	for idx, infile in enumerate(infiles):
		with h5.File(infile,'r') as f:
			for key in f.keys():
				# ignore the I3Index keys (not useful for our merged dataset)
				if key=='__I3Index__': continue
				# initialize the dictionary from the first input hdf5 file
				if idx == 0:
					dimensions[key] = 0
					dtypes[key]     = f[key].dtype
				else:
					assert key in dimensions, f'Dataset {infile} conatins key {key} which is missing from initial key dictionary'
					assert dtypes[key] == f[key].dtype, f'Different dtype from {dtypes[key]} for key {key}'
				dimensions[key] += f[key].shape[0]
		print(f'\rScanned file {idx} / {len(infiles)}', end='\r')

	print (f'Total number of rows for each object is as follows: {dimensions}')

	row_tracker = dict((key,0) for key in dimensions)

	#finally create the output file
	with h5.File(outfile_name,'w') as outfile:
		#create a dataset column for storing the merged filenames
		outfile.create_dataset('filename', (dimensions[PER_EVENT_COLUMN],), dtype=h5.special_dtype(vlen=bytes))
		#create the blank dataset columns for keys with total dimensions
		for key in dimensions:
			outfile.create_dataset(key, (dimensions[key],), dtype=dtypes[key])
		print (f'Output file structure created, filling with data now...')
		for infile in infiles:
			with h5.File(infile,'r') as f:
				n_events = f[PER_EVENT_COLUMN].shape[0]
				outfile['filename'][row_tracker[PER_EVENT_COLUMN] : row_tracker[PER_EVENT_COLUMN]+n_events] = bytes(infile, encoding='ASCII')
				for key in dimensions:
					print (f'\rCopying {key} from {infile}...             ', end='\r')
					size = f[key].shape[0]
					outfile[key][row_tracker[key] : row_tracker[key]+size] = f[key]
					row_tracker[key] += size
