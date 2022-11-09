#!/bin/bash
config_file_dir=$1
video_path=$2
source ~/.bashrc
conda activate DEEPLABCUT
module load cuda
videos=$video_path*.mov

for video in $videos;
do 
	echo $video
	echo $config_file_dir
	sbatch --mem=100G --partition=scu-gpu --gres=gpu:2 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu --wrap="python /home/dje4001/post_dlc/dlc_process.py --config_file_dir $config_file_dir --MP4_file_dir '$video'"

done


