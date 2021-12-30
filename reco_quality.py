#!/bin/python

'''
WARNING: Works on Trident simulation files

This script checks the muon reconstruction quality
L2 level reconstrucion is used to extract features for DOMs and used in GNN
quality check is required prior to use it for neural network
'''

import numpy as np
from icecube import icetray, dataio, dataclasses, simclasses, gulliver
from icecube import dataclasses as dc

#dictionary to use different reconstructions
reco_dict = {'MPEFit': ('MPEFit','MPEFitFitParams'), 'LineFit': ('LineFit','LineFitParams'),
		'SplineMPE': ('OnlineL2_SplineMPE', 'OnlineL2_SplineMPEFitParams')}


def get_true_muons(frame):
	'''
	This functions extracts the two true muon objects from I3MCTree
	Input: DAQ frame
	Output: tuple of two I3Particles
	'''
	mctree = frame['I3MCTree']
	prim = mctree.primaries[0]
	seco = mctree.get_daughters(prim)
	#first particle in the list of secndaries is the outgoing neutrino (for trident)
	return (seco[1],seco[2])
	
def get_TrueAverageTrackDir(frame):
	'''
	This function calculates the average track direction given 
	two true muon direction.
	Average direction is unweighted (i.e. independent of muon energies)
	Input: DAQ frame
	Output: I3Direction (averaged from two muon tracks)
	'''
	mminus,mplus = get_true_muons(frame)
	dir1 = mminus.dir
	dir2 = mplus.dir

	avg_x = 0.5*(dir1.x+dir2.x)
	avg_y = 0.5*(dir1.y+dir2.y)
	avg_z = 0.5*(dir1.z+dir2.z)

	#create I3Direction object
	avg_dir = dc.I3Direction(avg_x,avg_y,avg_z)
	#fill it with average direction values from two tracks
	#dc.I3Direction.set_theta_phi(avg_dir,avg_theta,avg_phi)
	return avg_dir

def get_RecoSTrackDir(frame,reco):
	'''
	This function gets the reconstructed track direction
	and the log likelihood value (for the purpose of checking
	goodness of the fit)
	Input: Physics Frame and type of reconstruction (string type)
	Output: Returns the tuple object of reconstructed I3Direction
	and log likelihood value
	'''
	global reco_dict
	reco_part = frame[reco_dict[reco][0]]
	reco_dir = reco_part.dir
	reco_logl = frame[reco_dict[reco][1]].rlogl
	return (reco_dir,reco_logl)

def opening_angle(dir1,dir2):
	'''
	Calculates the opening angle between two I3Direction objects
	'''
	th1,ph1=(dir1.theta,dir1.phi)
	th2,ph2=(dir2.theta,dir2.phi)

	dot=np.sin(th1)*np.cos(ph1)*np.sin(th2)*np.cos(ph2)+np.sin(th1)*np.sin(ph1)*np.sin(th2)*np.sin(ph2)+np.cos(th1)*np.cos(th2)

	return np.arccos(dot)

def get_dir_difference(frame,reco):
	'''
	This function calculates the difference between the true
	average muon track and the reconstructed track directions
	in terms of opening angle. It also calculates the opening
	angle between 1. maximum muon energy track and the reconstructed
	track and 2. maximum muon energy track and the average true
	track and then takes the difference between the two, to check
	if there is direction reconstruction bias due to energy 
	assymmetry between two muons.
	Input: Physics frame, reco type
	Output: tuple object containing angle differences and log
	likelihood value for the reconstruction
	'''

	avg_track = get_TrueAverageTrackDir(frame)
	rec_track, logval = get_RecoSTrackDir(frame,reco)

	mm,mp = get_true_muons(frame)
	if mm.energy>mp.energy:
		mxe_track = mm.dir
	else:
		mxe_track = mp.dir

	avg_rec_diff = opening_angle(avg_track,rec_track)
	mxe_avg_diff = opening_angle(avg_track,mxe_track)
	mxe_rec_diff = opening_angle(mxe_track,rec_track)

	return (avg_rec_diff,(mxe_rec_diff-mxe_avg_diff),logval)
