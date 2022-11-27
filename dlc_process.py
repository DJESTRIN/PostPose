#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process videos based on trained dlc model
"""
import argparse
import deeplabcut

def dlc_process(path_to_config, path_to_MP4):
    try:
        deeplabcut.analyze_videos(path_to_config,[path_to_MP4], auto_track=True,videotype='mp4')
        deeplabcut.create_video_with_all_detections(path_to_config, [path_to_MP4], videotype='mp4')
        deeplabcut.create_labeled_video(path_to_config,[path_to_MP4],color_by="individual",keypoints_only=False,
                                        trailpoints=10,draw_skeleton=True,track_method="ellipse")
    except:
        print("Post-process analysis failed")

parser=argparse.ArgumentParser()
parser.add_argument('--config_file_dir',type=str,required=True) #input to config file
parser.add_argument('--MP4_file_dir',type=str,required=True) #input to config file

if __name__=="__main__":
    args=parser.parse_args()
    dlc_process(args.config_file_dir,args.MP4_file_dir)
