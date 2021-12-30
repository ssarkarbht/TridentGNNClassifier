#!/bin/python

'''
Author: Sourav Sarkar
Date: November 27, 2021
Email: ssarkar1@ualberta.ca
Description: This script takes the h5 file and scales it's features using the
	mean and standard deviation for all the respective feature values
'''
node_features = [
            "ResidualTime", "PhotonTrackLength", "ChargeFirstPulse",
            "TrackDOMDistance", "TotalCharge", "TimeMaxPulse",
            "ChargeMaxPulse", "DeltaTimeStd"
        ]

graph_features = [
            "InitialTrackIntensity", "FinalTrackIntensity",
            "TrackSmoothness", "EventCharge", "EventTracklength"
        ]

import h5py as h5
import numpy as np
from optparse import OptionParser
#parser=OptionParser()
#parser.add_option("-f", "--Filename", dest="FNAME", type=int)

#(options, args) = parser.parse_args()
#filename = options.FNAME
#f = h5.File(filename,"a")

n_min = []
n_max = []
n_std = []
n_mean = []
g_min = []
g_max = []
g_std = []
g_mean = []

fileloc = "/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/dataset/resampled_dataset_8/dataset_split/"
f_train = h5.File(fileloc+"train.h5", "a")
f_val   = h5.File(fileloc+"val.h5", "a")
f_test  = h5.File(fileloc+"test.h5", "a")

for key in node_features:
	arr = []
	arr += list(f_train[key][:])
	arr += list(f_val[key][:])
	arr += list(f_test[key][:])

	arr = np.array(arr)

	minval = min(arr)
	maxval = max(arr)
	stdval = np.std(arr)
	meanval= np.mean(arr)
	print (f"Scale info: {key} : min:{minval}, max:{maxval}, std:{stdval}, mean:{meanval}")
	f_train[key][:] = (f_train[key][:]-meanval)/stdval
	f_val[key][:]   = (f_val[key][:]-meanval)/stdval
	f_test[key][:]  = (f_test[key][:]-meanval)/stdval
	minv = min([np.min(f_train[key][:]), np.min(f_val[key][:]),np.min(f_test[key][:])])
	maxv = max([np.max(f_train[key][:]), np.max(f_val[key][:]),np.max(f_test[key][:])])
	print (f"Post scaling range: {key} : min:{minv}, max:{maxv}")
	n_min.append(minval)
	n_max.append(maxval)
	n_std.append(stdval)
	n_mean.append(meanval)

for key in graph_features:
	arr = []
	arr += list(f_train[key][:])
	arr += list(f_val[key][:])
	arr += list(f_test[key][:])

	arr = np.array(arr)

	minval = min(arr)
	maxval = max(arr)
	stdval = np.std(arr)
	meanval= np.mean(arr)

	print (f"Scale info: {key} : min:{minval}, max:{maxval}, std:{stdval}, mean:{meanval}")
	f_train[key][:] = (f_train[key][:]-meanval)/stdval
	f_val[key][:]   = (f_val[key][:]-meanval)/stdval
	f_test[key][:]  = (f_test[key][:]-meanval)/stdval

	minv = min([np.min(f_train[key][:]), np.min(f_val[key][:]),np.min(f_test[key][:])])
	maxv = max([np.max(f_train[key][:]), np.max(f_val[key][:]),np.max(f_test[key][:])])
	print (f"Post scaling range: {key} : min:{minv}, max:{maxv}")
	g_min.append(minval)
	g_max.append(maxval)
	g_std.append(stdval)
	g_mean.append(meanval)

print (n_min,n_max,n_std,n_mean)
print (g_min,g_max,g_std,g_mean)
g11 = f_train.create_group('NodeScaleInfo')
g12 = f_train.create_group('GraphScaleInfo')

g21 = f_val.create_group('NodeScaleInfo')
g22 = f_val.create_group('GraphScaleInfo')

g31 = f_test.create_group('NodeScaleInfo')
g32 = f_test.create_group('GraphScaleInfo')

g11.create_dataset('MinRange', data=n_min)
g11.create_dataset('MaxRange', data=n_max)
g11.create_dataset('StdValue', data=n_std)
g11.create_dataset('MeanValue', data=n_mean)

g12.create_dataset('MinRange', data=g_min)
g12.create_dataset('MaxRange', data=g_max)
g12.create_dataset('StdValue', data=g_std)
g12.create_dataset('MeanValue', data=g_mean)

g21.create_dataset('MinRange', data=n_min)
g21.create_dataset('MaxRange', data=n_max)
g21.create_dataset('StdValue', data=n_std)
g21.create_dataset('MeanValue', data=n_mean)

g22.create_dataset('MinRange', data=g_min)
g22.create_dataset('MaxRange', data=g_max)
g22.create_dataset('StdValue', data=g_std)
g22.create_dataset('MeanValue', data=g_mean)

g31.create_dataset('MinRange', data=n_min)
g31.create_dataset('MaxRange', data=n_max)
g31.create_dataset('StdValue', data=n_std)
g31.create_dataset('MeanValue', data=n_mean)

g32.create_dataset('MinRange', data=g_min)
g32.create_dataset('MaxRange', data=g_max)
g32.create_dataset('StdValue', data=g_std)
g32.create_dataset('MeanValue', data=g_mean)

f_train.close()
f_val.close()
f_test.close()

