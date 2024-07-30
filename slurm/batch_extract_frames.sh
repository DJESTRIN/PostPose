#!/bin/bash
folderoh=/athena/listonlab/scratch/dje4001/mdt02/tmt_dlc_class/TMT-Estrin-2023-08-29/videos/
for directory in $(find $folderoh -name '*mp4')
do
folder="${directory##*/}"
filename=${folder::-4}
dropdir="/athena/listonlab/scratch/dje4001/mdt02/tmt_dlc_class/TMT-Estrin-2023-08-29/labeled-data/$filename/"
sbatch --job-name=$filename --mem=10G --partition=scu-cpu,sackler-cpu,sackler-gpu,scu-gpu --wrap="/home/dje4001/post_dlc/custom_frame_extraction.sh $directory $dropdir"
done
