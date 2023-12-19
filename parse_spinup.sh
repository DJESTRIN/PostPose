#!/bin/bash
# called during srun with screen!
source ~/.bashrc
conda activate spyder-env
h5files=$1*.h5

for h5_file in $h5files;
do
	echo $h5_file
	python ~/post_dlc/parse.py --h5_dir $h5_file

done
