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


getfileloc = "/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/dataset/resampled_dataset_8/dataset_split/"
gfile = h5.File(getfileloc+"train.h5","r")
nscale = gfile["NodeScaleInfo"]
nmean= nscale["MeanValue"]
nstd  = nscale["StdValue"]
gscale = gfile["GraphScaleInfo"]
gmean = gscale["MeanValue"]
gstd = gscale["StdValue"]

fileloc="/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/dataset/weight_dataset_10/dataset_split/copy/"
f_train = h5.File(fileloc+"train.h5", "a")

for i,key in enumerate(node_features):
	stdval = nstd[:][i]
	meanval= nmean[:][i]
	print (stdval,meanval)
	f_train[key][:] = (f_train[key][:]-meanval)/stdval
	minv = np.min(f_train[key][:])
	maxv = np.max(f_train[key][:])
	print (f"Post scaling range: {key} : min:{minv}, max:{maxv}")

for i,key in enumerate(graph_features):
	stdval = gstd[:][i]
	meanval=gmean[:][i]

	print (stdval,meanval)
	f_train[key][:] = (f_train[key][:]-meanval)/stdval

	minv = np.min(f_train[key][:])
	maxv = np.max(f_train[key][:])
	print (f"Post scaling range: {key} : min:{minv}, max:{maxv}")


#g11 = f_train.create_group('NodeScaleInfo')
#g12 = f_train.create_group('GraphScaleInfo')



f_train.close()

