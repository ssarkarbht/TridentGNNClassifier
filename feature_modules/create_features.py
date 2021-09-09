#!/bin/python
'''
Author: Sourav Sarkar
Date: May 3, 2021
Email: ssarkar1@ualberta.ca
Objective: This script takes the L2 level eventfiles
	and extrtacts the relevant information to
	prepare dataset for the graph neural network,
	in addition it also stores the metadata needed
	to track additonal event information.
'''

#Import necessary modules
from icecube import icetray, dataio, phys_services
from icecube import dataclasses as dc
from I3Tray import I3Tray
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import sys
import tables
import numpy as np


#load the GCD file for extracting DOM geometry
f_geo = dataio.I3File('/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/feature_modules/GeoCalibDetectorStatus_2016.57531_V0.i3.gz')
geo_frame = f_geo.pop_frame(icetray.I3Frame.Geometry)
geo = geo_frame['I3Geometry']


#Following class extracts the data from each frame in an i3 file
class extract_data():
	'''
	Input: frame from an I3 file and geometry object
	'''
	def __init__(self, frame, geo, reco='OnlineL2_SplineMPE', pulses='SRTInIcePulses', charge_threshold=0.5):
		#List of keys for the DOM properties dictionary
		self.vertex_features=['ResidualTime', 
                'PhotonTrackLength', 'ChargeFirstPulse', 'TrackDOMDistance',
                'TotalCharge', 'TimeMaxPulse', 'ChargeMaxPulse', 'DeltaTimeStd']
		#Load the L2 track reconstruction
		self.track_reco = frame[reco]
		#Set the track as infinite track (for residual time calculation purpose)
		self.track_reco.shape = dc.I3Particle.InfiniteTrack
		#extract the DOM pulses
		#hit_obj = frame["SRTInIcePulses"]
		hit_obj = frame[pulses]
		self.dom_hits = hit_obj.apply(frame)
		#Get the timeshift value for time offset calculation
		self.time_shift = frame['TimeShift'].value
		# Create a dictionary map for DOMs and it corresponding coordinates
		self.dom_pos = {}
		for i in self.dom_hits.keys():
#			self.dom_pos[i] = dc.I3Position(dom_positions[i]['x'],dom_positions[i]['y'],dom_positions[i]['z'])
			self.dom_pos[i] = geo.omgeo.get(i).position
		#Set the charge threshold for each DOM pulse to be considered
		self.charge_threshold = charge_threshold

	def dom_residual_time(self, omkey, mincut=-800., maxcut=5000.):
		'''
		takes the DOM and extracts the time residual value for the DOM
		Time residual: time difference between the first DOM hit time and expected
			cherenkov photon arrival time.
		Parameters:
		omkey: I3OMKey
		mincut: minimum allowed Time residual value to be registered
		maxcut: maximum allowed time residual value to be registered
		'''
		dom_position = self.dom_pos[omkey]
		#Calculate the expected cherenkov photon arrival time
		cherenkov_time = self.track_reco.time+self.time_shift+phys_services.I3Calculator.cherenkov_time(self.track_reco, dom_position)
		#Get all the hits for the given DOM
		domhit = self.dom_hits.get(omkey)
		#Boolean tracker for detecting the first hit
		domhit_bool = False
		#Loop through the hits to get the first significant hit time
		for hit in domhit:
			if hit.charge<self.charge_threshold: continue
			domhit_time = hit.time+self.time_shift
			res_time    = domhit_time-cherenkov_time
			if not(mincut<=res_time<=maxcut): continue
			domhit_bool = True
			self.time_threshold=domhit_time
			break
		#If no significant hit is found return None
		if not(domhit_bool): return None
		#if not(mincut<=res_time<=maxcut): return None
		#Else return the first time for the DOM
		return res_time

	def dom_photon_tracklength(self, omkey):
		'''
		This function calculates the length along the particle track (starting
		from the track vertex reconstruction) where the cherenkov photon for 
		given DOM was generated on the track - it is measurement for possible
		correlation between two track separation as function of track distance
		from vertex/time.
		'''
		#Calculate the cherenkove photon position on the track
		cherenkov_position = phys_services.I3Calculator.cherenkov_position(self.track_reco,self.dom_pos[omkey])
		# get the track vertex position from the reconstruction
		vertex_position = self.track_reco.pos
		# get the parametric value (equivalent of length) of a line from two points
		length_param = (cherenkov_position.x-vertex_position.x)/self.track_reco.dir.x
		return length_param

	def dom_cherenkov_distance(self, omkey):
		'''
		This function calculates the length between the cherenkov photon
		generation position on the track and the photon hit position (i.e. the
		DOM position) - it is a measurement of how far the DOM is from the track
		(if it is susceptible to large scattering)
		'''
		cherenkov_dist = phys_services.I3Calculator.cherenkov_distance(self.track_reco,self.dom_pos[omkey])
		return cherenkov_dist

	def dom_pulse_properties(self, omkey):
		'''
		This function calculates/extracts some of the DOM hit properties such
		as DOM hit times, DOM hit charges, the amount of spread in all the 
		hit times in a DOM
		'''
		# get all the pulses for the given DOM
		pulses = self.dom_hits.get(omkey)
		# Create a blank list of charge and time
		charge=[]
		time=[]
		#Loop through the pulses
		for p in pulses:
			charge.append(p.charge)
			time.append(p.time+self.time_shift)
		charge = np.array(charge)
		time = np.array(time)
		#Charge amount of the first pulse
		first_charge = charge[np.where((charge>=self.charge_threshold) & (time>=self.time_threshold))][0]
		#Total amount of charge seen by the given DOM
		total_charge = np.sum(charge)
		#Charge and time of the maximum charge pulse
		max_charge = charge.max()
		max_time   = time[charge.argmax()]
		#standard deviation of the time differences between two consecutive pulse
		time_difference     = np.diff(time)
		if len(time_difference)==0: time_difference_std = 0.
		else: time_difference_std = np.std(time_difference)
		return (first_charge, total_charge, max_charge, max_time, time_difference_std)

	def dom_edge_information(self,omkey):
		'''
		This function calculates the two parameters needed to construct
		edge information in the graph structure between two DOMs.
		Parameter 1: Closest approach position of the given DOM on track
		Parameter 2: Direction of the track-to-DOM distance vector
		'''
		edge_pos = phys_services.I3Calculator.closest_approach_position(self.track_reco,self.dom_pos[omkey])
		
		direction = self.dom_pos[omkey]-edge_pos
		normed_dir = dc.I3Direction(direction.x,direction.y,direction.z)
		return (edge_pos,normed_dir)

	def get_frame_data(self, feature_scale=False):
		'''
		Driver function that calls to the above functions for a frame and loops
		throug all the hit DOMs and extracts the info for each DOM and stores in
		disctionaries.
		'''
		features = {feature: [] for feature in self.vertex_features}
		edges    = {'EdgePositionX': [], 'EdgePositionY': [], 'EdgePositionZ': [],
                            'EdgeDirectionX': [], 'EdgeDirectionY': [], 'EdgeDirectionZ': [],
                            'EdgeCoordinateX': [], 'EdgeCoordinateY': [], 'EdgeCoordinateZ':[]}
		omkeys   = []

		#loop through all the DOM hits
		for omkey in self.dom_hits.keys():
			#if no residual time found, skip rest of the extraction process
			if self.dom_residual_time(omkey)==None: continue
			features['ResidualTime'].append(self.dom_residual_time(omkey))
			features['PhotonTrackLength'].append(self.dom_photon_tracklength(omkey))
			features['TrackDOMDistance'].append(self.dom_cherenkov_distance(omkey))
			props = self.dom_pulse_properties(omkey)
			features['ChargeFirstPulse'].append(props[0])
			features['TotalCharge'].append(props[1])
			features['ChargeMaxPulse'].append(props[2])
			features['TimeMaxPulse'].append(props[3])
			features['DeltaTimeStd'].append(props[4])

			eposition, edirection = self.dom_edge_information(omkey)
			edges['EdgePositionX'].append(eposition.x)
			edges['EdgePositionY'].append(eposition.y)
			edges['EdgePositionZ'].append(eposition.z)
			edges['EdgeDirectionX'].append(edirection.x)
			edges['EdgeDirectionY'].append(edirection.y)
			edges['EdgeDirectionZ'].append(edirection.z)

			edges['EdgeCoordinateX'].append(self.dom_pos[omkey].x)
			edges['EdgeCoordinateY'].append(self.dom_pos[omkey].y)
			edges['EdgeCoordinateZ'].append(self.dom_pos[omkey].z)
			assert omkey not in omkeys
			omkeys.append(omkey)
		if feature_scale and len(features['ResidualTime'])!=0:
			total_tracklength = max(features['PhotonTrackLength']) - min(features['PhotonTrackLength'])
			features['PhotonTrackLength'] = list(np.array(features['PhotonTrackLength'])/(total_tracklength+1e-8) - 0.5)
		return features, edges, np.array(omkeys)

def process_frame(frame, interaction_type=-1, runid=-1):
	#check if the physics frame is InIceSplit
	if frame['I3EventHeader'].sub_event_stream!='InIceSplit': return False

	#Check if the frame has the desired reconstruction
	if not(frame.Has('OnlineL2_SplineMPE')): return False

	# Store metadata info for ground truth and analysis

	#Primary particle Truth info
	frame['InteractionType']= dc.I3Double(interaction_type)
	nu = frame['I3MCTree_postMuonProp'].get_primaries()[0]
	frame['InteractionVertex'] = dc.I3VectorFloat([nu.pos.x,nu.pos.y,nu.pos.z])

	frame['NeutrinoEnergy'] = dc.I3Double(nu.energy)
	frame['NeutrinoPDGCode']= dc.I3Double(nu.pdg_encoding)
	frame['NeutrinoAzimuth']= dc.I3Double(nu.dir.azimuth)
	frame['NeutrinoZenith'] = dc.I3Double(nu.dir.zenith)

	#Secondary particle Truth info
	secondaries = frame['I3MCTree_postMuonProp'].get_daughters(nu)
	if interaction_type==1:
		MuMinus     = secondaries[1]
		MuPlus      = secondaries[2]
		frame['TrackEnergy'] = dc.I3VectorFloat([MuMinus.energy, MuPlus.energy])
		frame['TrackPDGCode']= dc.I3VectorFloat([MuMinus.pdg_encoding, MuPlus.pdg_encoding])
		frame['TrackAzimuth']= dc.I3VectorFloat([MuMinus.dir.azimuth, MuPlus.dir.azimuth])
		frame['TrackZenith'] = dc.I3VectorFloat([MuMinus.dir.zenith, MuPlus.dir.zenith])
		frame['TrackLength'] = dc.I3VectorFloat([MuMinus.length, MuPlus.length])
		frame['HadronEnergy']= dc.I3Double(secondaries[3].energy) 

	elif interaction_type==2:
		Mu          = secondaries[0]
		frame['TrackEnergy'] = dc.I3VectorFloat([Mu.energy,0])
		frame['TrackPDGCode']= dc.I3VectorFloat([Mu.pdg_encoding,0])
		frame['TrackAzimuth']= dc.I3VectorFloat([Mu.dir.azimuth,0])
		frame['TrackZenith'] = dc.I3VectorFloat([Mu.dir.zenith,0])
		frame['TrackLength'] = dc.I3VectorFloat([Mu.length,0])
		frame['HadronEnergy']= dc.I3Double(secondaries[1].energy)

	frame['RunID']  = icetray.I3Int(runid)
	frame['EventID'] = icetray.I3Int(frame['I3EventHeader'].event_id)

	track_reco = frame['OnlineL2_SplineMPE']
	frame['RecoX'] = dc.I3Double(track_reco.pos.x)
	frame['RecoY'] = dc.I3Double(track_reco.pos.y)
	frame['RecoZ'] = dc.I3Double(track_reco.pos.z)
	frame['RecoAzimuth'] = dc.I3Double(track_reco.dir.azimuth)
	frame['RecoZenith']  = dc.I3Double(track_reco.dir.zenith)

	extract = extract_data(frame,geo)
	features, edges, _ = extract.get_frame_data()

	for f_key in features.keys():
		frame[f_key] = dc.I3VectorFloat(features[f_key])
	for e_key in edges.keys():
		frame[e_key] = dc.I3VectorFloat(edges[e_key])

	frame['VertexNumber'] = icetray.I3Int(len(features[list(features.keys())[0]]))
#	frame['OneWeight']    = dc.I3Double(frame['I3MCWeightDict']['OneWeight'])
#	frame['NEvents']      = icetray.I3Int(frame['I3MCWeightDict']['NEvents'])
	return True

def create_dataset(infiles, outfile, int_type, rid):
	vertex_features = ['ResidualTime',
                'PhotonTrackLength', 'ChargeFirstPulse', 'TrackDOMDistance',
                'TotalCharge', 'TimeMaxPulse', 'ChargeMaxPulse', 'DeltaTimeStd']

	edge_info = ['EdgePositionX', 'EdgePositionY', 'EdgePositionZ',
                    'EdgeDirectionX', 'EdgeDirectionY', 'EdgeDirectionZ',
                    'EdgeCoordinateX', 'EdgeCoordinateY', 'EdgeCoordinateZ']

	metadata_keys = ['InteractionType', 'InteractionVertex',
                   'NeutrinoEnergy', 'NeutrinoPDGCode', 'NeutrinoAzimuth', 'NeutrinoZenith',
                   'TrackEnergy', 'TrackPDGCode', 'TrackAzimuth', 'TrackZenith', 'TrackLength',
                   'RunID', 'EventID', 'RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith',
                   'VertexNumber', 'HadronEnergy']#,'OneWeight','NEvents']

	tray = I3Tray()
	tray.AddModule('I3Reader', FilenameList=infiles)
	tray.AddModule(process_frame, 'process_frame', interaction_type=int_type,
                       runid=rid, Streams=[icetray.I3Frame.Physics])
	tray.AddModule(I3TableWriter, 'I3TableWriter', keys=vertex_features+edge_info+metadata_keys,
                            TableService=I3HDFTableService(outfile),
                            SubEventStreams=['InIceSplit'],
                            BookEverything=False)
	tray.Execute()
	tray.Finish()

if __name__=='__main__':
	from optparse import OptionParser
	parser=OptionParser()
	parser.add_option("-b", "--BatchNumber", dest="BATCH", type=int)
	parser.add_option("-d", "--DatasetNumber", dest="DSET", type=int)
	parser.add_option("-i", "--InteractionType", dest="ITYPE", type=int)
	parser.add_option("-p", "--ParticleType", dest="PTYPE", type=int)
	parser.add_option("-f", "--FileLocation", dest="FLOC", type=str)
	parser.add_option("-o", "--OutputLocation", dest="OUT", type=str)

	(options, args) = parser.parse_args()

	dset_dict = {1: 'dataset_01', 2: 'dataset_02', 3: 'dataset_03',
         4: 'dataset_04', 5: 'dataset_05', 6: 'dataset_06',
        7: 'dataset_07', 8: 'dataset_08', 9: 'dataset_09',
        10: 'dataset_10'}

	infile_path = options.FLOC
	outfile_path = options.OUT
	rid = int(options.ITYPE*1e6+options.PTYPE*1e5+1e3+options.DSET*1e1+options.BATCH)
	if options.ITYPE==1 and options.PTYPE==1:
		fileloc = dset_dict[options.DSET]+'/L2File/'
		filename = 'NumuEvent_L2_D'+'{:02d}'.format(options.DSET)+'_B'+'{:1d}'.format(options.BATCH)+'_preselection.i3'
		filename = fileloc+filename
		outfile = 'NumuEvent_L2_D'+'{:02d}'.format(options.DSET)+'_B'+'{:1d}'.format(options.BATCH)+'_resampled.h5'

	elif options.ITYPE==2 and options.PTYPE==1:
		fileloc = dset_dict[options.DSET]+'/L2Files/'
		filename = 'Numu_CC_L2_D'+'{:02d}'.format(options.DSET)+'_B'+'{:1d}'.format(options.BATCH)+'_preselection.i3'
		filename = fileloc+filename
		outfile = 'Numu_CC_L2_D'+'{:02d}'.format(options.DSET)+'_B'+'{:1d}'.format(options.BATCH)+ '_resampled.h5'

	elif options.ITYPE==2 and options.PTYPE==2:
		fileloc = dset_dict[options.DSET]+'/L2Files/'
		filename = 'NumuBar_CC_L2_D'+'{:02d}'.format(options.DSET)+'_B'+'{:1d}'.format(options.BATCH)+'_preselection.i3'
		filename = fileloc+filename
		outfile = 'NumuBar_CC_L2_D'+'{:02d}'.format(options.DSET)+'_B'+'{:1d}'.format(options.BATCH)+'_resampled.h5'

#	infile_path = '/data/user/ssarkar/TridentProduction/simulation/datasim/numu/run01/dataset_01/L2Files/'
#	infile_path = '/data/user/ssarkar/TridentProduction/simulation/datasim/resampled_numu/run01/dataset_01/L2File/'
#	infile_path = '/data/user/ssarkar/TridentProduction/simulation/datasim/cc_events/run01/dataset_01/L2Files/'
#	infile_path = '/data/user/ssarkar/TridentProduction/simulation/datasim/resample_cc/run01/dataset_01/L2Files/'
#	filename = 'NumuEvent_L2_D01_B1_resampled.i3'
#	filename = 'Numu_CC_L2_D01_B1_preselection.i3'
#	filename = 'NumuEvent_CC_L2_D01_B5.i3'
#	outfile_path = '/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/dataset/resampled_dataset_3/'
#	outfile = 'NumuEvent_CC_L2_D01_B5.h5'
#	outfile = 'Numu_CC_L2_D01_B1_resampled.h5'
#	outfile = 'NumuEvent_L2_D01_B1_resampled.h5'
	print ("Creating dataset with following details: ")
	print (f"Infile: {infile_path+filename} \n Outfile: {outfile_path+outfile} \n RunID: {rid}")
	create_dataset([infile_path+filename],outfile_path+outfile,options.ITYPE,rid)
