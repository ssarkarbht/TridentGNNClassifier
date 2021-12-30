#!/bin/python

from create_features import *
from glob import glob

tags=['120101', '120102', '120103']
for tagname in tags:
	if tagname=='120101': batch=['1','3','4','5']
	elif tagname=='120102': batch=['1','2','3','4','5']
	elif tagname=='120103': batch=['1','2','3']
	for i in batch:
		tag=tagname+i
		print (f'Processing tag {tag}')
		filelist=sorted(glob("/data/user/ssarkar/TridentProduction/datasets/trident/07_L2Files/tmp/ww_newgeo1_numu_"+tag+"*"))
#filelist=['full_trident.i3']
		if tag=='1201023':
			filelist.remove('/data/user/ssarkar/TridentProduction/datasets/trident/07_L2Files/tmp/ww_newgeo1_numu_1201023_02_043_out.i3')
		outfile = "/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/dataset/weight_dataset_10/trident_"+tag+".h5"
		create_dataset(filelist,outfile,1,int(tag), book_eventfeature=True, book_weight=True)
