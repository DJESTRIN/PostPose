#!/bin/bash
config_file_dir=$1
video_path=$2
file_extension=$3
videos=$video_path*.$file_extension

for video in $videos;
do 
	echo $video
	echo $config_file_dir
	sbatch --mem=200G --partition=scu-cpu,sackler-cpu --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu --wrap="bash /home/dje4001/post_dlc/init_sbatch_parse.sh $config_file_dir $video"

done


