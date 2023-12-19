#!/bin/bash
input_dir=/cephfs/dje4001/test_video_convert/

cd $input_dir
for video in $(find $input_dir -type f -name "*.h264")
do	
	output_name="${video/h264/mp4}"
	echo $output_name is being converted
	#ffmpeg -hwaccel cuda -i "$video" "$output_name"
	ffmpeg -hwaccel cuvid -i "$video" -c:v h264_nvenc -preset slow "$output_name"
done
