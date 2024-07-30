#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEEPLABCUT: create project
"""
import os
import deeplabcut

# Get video files 
video_list=[]
for root, dirs, files in os.walk("/athena/listonlab/store/dje4001/rsync_data/BB0046_RECORDING_SYSTEM/"):
    for file in files:
        if file.endswith(".mp4"):
             video_list.append(str(os.path.join(root, file)))

#video_dirs=','.join(video_list)
#video_dirs=video_dirs.split(',')

#config_path=deeplabcut.create_new_project('BB0046_TMT','DavidEstrin',video_list,working_directory="/athena/listonlab/scratch/dje4001/deeplabcut_David/",copy_videos=True,multianimal=False)
config_path="/athena/listonlab/scratch/dje4001/deeplabcut_David/BB0046_TMT-DavidEstrin-2022-11-17/config.yaml"
deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False,crop=False)

