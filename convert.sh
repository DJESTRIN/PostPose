#!/bin/bash
input_dir=$1
source ~/.bashrc
conda activate peri_dlc

cd $input_dir
for video in $(find $input_dir -type f -name "*.h264")
do	
	output_name="${video/h264/mp4}"
	echo $output_name is being converted
	ffmpeg -n -framerate 10 -i $video -c copy $output_name
done
