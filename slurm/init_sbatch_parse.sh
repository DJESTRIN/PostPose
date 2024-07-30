#!/bin/bash
config_file_dir=$1
video=$2
source ~/.bashrc
conda activate DEEPLABCUT_GUI
module load cuda
module load gcc-8.2.0-gcc-4.8.5-7ox3vie
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo $video
python /home/dje4001/post_dlc/parse_dlc_tracking.py --config_file_dir $config_file_dir --MP4_file_dir $video
