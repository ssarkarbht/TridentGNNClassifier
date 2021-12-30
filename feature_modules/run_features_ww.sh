#!/bin/bash

#eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
#i3_env=/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh

run_script=/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/feature_modules/create_features.py

infileloc=/data/user/ssarkar/TridentProduction/simulation/datasim/cc_events/run01/
outfileloc=/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/dataset/weight_dataset_10/
itype=2

#infileloc=/data/user/ssarkar/TridentProduction/simulation/datasim/resampled_numu/run01/
#outfileloc=/data/user/ssarkar/TridentProduction/reconstruction/trident_gnn/dataset/resampled_dataset_9/
#itype=1


#for i in {1..2}
#do
i=2
j=1
for k in {1..5}
do
	echo "PType: " $i "Dataset Number: " $j "Batch Number: " $k
	python $run_script -b $k -d $j -i $itype -p $i -f $infileloc -o $outfileloc
	wait
done
