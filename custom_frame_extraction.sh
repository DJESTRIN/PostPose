#!/bin/bash
input=$1
output=$2
source ~/.bashrc
conda activate spyder-env
python ~/post_dlc/custom_frame_extraction.py --input_video_dir $input --output_image_dir $output
