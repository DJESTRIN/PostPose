#!/bin/bash
input_dir=$1
source ~/.bashrc
conda activate peri_dlc

cd $input_dir
for video in $(find $input_dir -type f -name "*.mp4")
do
	fn=$(basename $video)
	dn=$(dirname $video)
	folder_name=${dn##*/}
	output_name="${dn}/$folder_name${fn}"
	echo $output_name
	mv "$video" "$output_name"
done


