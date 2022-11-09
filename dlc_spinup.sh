#!/bin/bash
video_dir_start=$1
code_directory=/home/dje4001/post_dlc/
scratch=/athena/listonlab/scratch/dje4001/deeplabcut/
storage=/scu-storage03/listonlab/scratch/dje4001/deeplabcut/

mkdir -p $scratch
mkdir -p $storage

source ~/.bashrc
conda activate DEEPLABCUT
module load cuda

# Move video folder to scratch in 1 job
copy_jid=$(sbatch --mem=300G --partition=scu-cpu --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu --wrap="bash $code_directory/dlc_to_scratch.sh '$video_dir_start' '$scratch'")


# Get FULL path to config file
cd $scratch
config_file_dir=$(find ~+ -maxdepth 2 -type f -name "config.yaml")


# Run training in 1 job
training_jid=$(sbatch --dependency=afterany:$copy_jid --mem=300G --partition=scu-gpu --gres=gpu:2 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu --wrap="python $code_directory/train_dlc.py --config_file_dir $config_file_dir"

# Run post-processing: will run MANY jobs
sbatch --dependency=afterany:$training_jid --job-name=dlc_process --mem=5G --partition=scu-cpu --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu --wrap="bash $code_directory/process_dlc.sh $config_file_dir $scratch"

# Send data to midtier, concat output?
sbatch --job-name=dlc_process --dependency=singleton --mem=50G --partition=scu-cpu --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu --wrap="bash $code_directory/spin_down_dlc.sh $scratch"
