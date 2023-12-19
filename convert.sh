#!/bin/bash
input_dir=/athena/listonlab/store/dje4001/bb0046_temp_store/bb0046/

cd $input_dir
for dir in $input_dir*/
do	
	echo $dir
	sbatch --job-name=convertvideo --mem=200G --partition=scu-cpu,sackler-cpu --wrap="bash ~/post_dlc/convert_s.sh $dir"
done
