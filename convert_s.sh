#!/bin/bash
input_dir=$1
cd $input_dir
source ~/.bashrc
conda activate peri_dlc

for video in $(find $input_dir -type f -name "*.h264")
do	
	output_name="${video/h264/mp4}"
	echo $output_name is being converted
	ffmpeg -framerate 20 -i $video -c copy $output_name
done
